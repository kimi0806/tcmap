from flask import Flask, render_template, request, jsonify
import requests
from shapely.geometry import shape
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

WFS_URL = (
    "https://wfs.cartografia.agenziaentrate.gov.it"
    "/inspire/wfs/owfs01.php"
)

HEADERS = {
    "User-Agent": "tcfolder-web/1.0"
}

TIMEOUT = 60


# Reuse TCP/TLS connections across requests instead of
# opening a brand new connection for every single tile.
SESSION = requests.Session()


# Tile fetches are I/O-bound (waiting on the government
# WFS server), so we fire them off concurrently instead
# of one-at-a-time. Kept modest to stay a good citizen of
# a public government endpoint.
WFS_MAX_WORKERS = 12


# The Agenzia Entrate WFS works reliably with small BBOX
# requests.
WFS_TILE_SIZE = 0.005


# Safety limit.
WFS_MAX_TILES = 100


# =========================================================
# FOGLIO LOOKUP CACHE
#
# Locating a Foglio's zoning polygon (and the parcels
# inside it) requires scanning many WFS tiles. A Foglio's
# boundaries and parcels essentially never change, so once
# we've found one we keep it in memory for the life of the
# process instead of re-scanning the whole Comune on every
# search.
# =========================================================

_cache_lock = threading.Lock()
_foglio_zoning_cache = {}
_foglio_parcels_cache = {}
_buildings_cache = {}


def _cache_get(cache, key):
    with _cache_lock:
        return cache.get(key)


def _cache_set(cache, key, value):
    with _cache_lock:
        cache[key] = value


# GML coordinate reference system.
GML_SRS = "urn:ogc:def:crs:EPSG::6706"


PARCEL_LAYER = "CP:CadastralParcel"


PARCEL_FIELDS = {
    "INSPIREID_LOCALID",
    "INSPIREID_NAMESPACE",
    "LABEL",
    "NATIONALCADASTRALREFERENCE",
    "ADMINISTRATIVEUNIT",
}


# =========================================================
# BUILDING FOOTPRINTS (fabbricati)
#
# The Agenzia Entrate cadastral WFS only exposes Parcels
# and Zoning (confirmed via GetCapabilities) - no building
# layer. OpenStreetMap has good building-footprint coverage
# in Italian municipalities (often sourced from the Comune
# itself), so we use its public Overpass API instead, and
# clip results to the parcel polygon so we only show
# buildings genuinely inside the searched Particella.
# =========================================================

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OVERPASS_TIMEOUT = 25


# =========================================================
# COMUNE → BELFIORE CODE
# =========================================================

COMUNI = {
    "BELMONTE MEZZAGNO": "A764",
    "PALERMO": "G273",
}


# Approximate EPSG:6706 / WGS84 municipality BBOX.
#
# Order:
# south, west, north, east
#
# These are fallback areas only.
COMUNE_BBOXES = {

    "A764":
        "37.9300,13.3300,38.0850,13.4650",

    "G273":
        "38.0450,13.2350,38.2350,13.4550"

}


# =========================================================
# HELPERS
# =========================================================

def local_name(tag):

    return (
        tag.rsplit("}", 1)[-1]
        if "}" in tag
        else tag
    )


def normalize_value(value):

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .upper()
    )


def normalize_number(value):

    """
    Normalize cadastral numbers.

    Examples:

        "001" → "1"
        "0001" → "1"
        "1" → "1"

    This is useful because cadastral systems may
    represent the same number with leading zeros.
    """

    value = normalize_value(value)

    if not value:
        return ""

    # Keep only the leading numeric part when the
    # value is purely numeric.
    if re.fullmatch(r"0*\d+", value):

        try:
            return str(int(value))
        except ValueError:
            pass

    return value


def parse_bbox(value):

    if not value:

        raise ValueError(
            "Missing bbox."
        )


    parts = [
        part.strip()
        for part in value.split(",")
    ]


    if len(parts) != 4:

        raise ValueError(
            "BBOX must contain 4 comma-separated numbers."
        )


    south, west, north, east = [
        float(part)
        for part in parts
    ]


    if south >= north or west >= east:

        raise ValueError(
            "BBOX order must be "
            "south,west,north,east."
        )


    return (
        f"{south},"
        f"{west},"
        f"{north},"
        f"{east}"
    )


def get_comune_code(comune):

    if not comune:
        return None


    stripped = (
        comune
        .strip()
        .upper()
    )


    # User may already provide the Belfiore code.
    if stripped in COMUNE_BBOXES:

        return stripped


    # Belfiore format.
    if re.fullmatch(
        r"[A-Z][0-9]{3}",
        stripped
    ):

        return stripped


    normalized = " ".join(
        stripped.split()
    )


    return COMUNI.get(
        normalized
    )


def get_search_bbox(
    raw_bbox,
    comune_code
):

    if raw_bbox:

        return (
            parse_bbox(raw_bbox),
            "map"
        )


    fallback = COMUNE_BBOXES.get(
        comune_code
    )


    if fallback:

        return (
            parse_bbox(fallback),
            "comune"
        )


    raise ValueError(
        "Missing bbox."
    )


# =========================================================
# CADASTRAL REFERENCE VARIANTS
# =========================================================

def build_reference_variants(
    comune_code,
    foglio,
    particella
):

    variants = set()


    if not comune_code:
        return variants


    if not foglio:
        return variants


    if not particella:
        return variants


    try:

        foglio_number = int(
            foglio
        )


        particella_number = (
            normalize_number(
                particella
            )
        )


        padded_foglio = (
            f"{foglio_number:04d}"
        )


        # -------------------------------------------------
        # Format observed in Agenzia Entrate WFS.
        #
        # Normal parcel:
        # G273_0033A0.174
        #
        # Cadastral road:
        # G273_0033A0.STRADA001
        #
        # Some Fogli use B0 instead of A0.
        # -------------------------------------------------

        for zoning_suffix in ("A0", "B0"):

            variants.add(
                f"{comune_code}_"
                f"{padded_foglio}"
                f"{zoning_suffix}."
                f"{particella_number}"
            )


            variants.add(
                f"IT.AGE.PLA."
                f"{comune_code}_"
                f"{padded_foglio}"
                f"{zoning_suffix}."
                f"{particella_number}"
            )


        # -------------------------------------------------
        # Previous fallback format.
        # -------------------------------------------------

        variants.add(
            f"{comune_code}_"
            f"{padded_foglio}"
            f"00."
            f"{particella_number}"
        )


        variants.add(
            f"IT.AGE.PLA."
            f"{comune_code}_"
            f"{padded_foglio}"
            f"00."
            f"{particella_number}"
        )


        # -------------------------------------------------
        # Simple fallback:
        #
        # G273.141.578
        # -------------------------------------------------

        variants.add(
            f"{comune_code}."
            f"{foglio_number}."
            f"{particella_number}"
        )


    except ValueError:

        pass


    return variants


# =========================================================
# GML COORDINATES
# =========================================================

def pairs_from_poslist(text):

    values = [
        float(value)
        for value in text.split()
    ]


    pairs = []


    for index in range(
        0,
        len(values) - 1,
        2
    ):

        lat = values[index]
        lon = values[index + 1]


        # GeoJSON uses:
        #
        # [longitude, latitude]

        pairs.append([
            lon,
            lat
        ])


    if (
        pairs
        and pairs[0] != pairs[-1]
    ):

        pairs.append(
            pairs[0]
        )


    return pairs


# =========================================================
# EXTRACT CADASTRAL PARCELS FROM GML
# =========================================================

def extract_parcels(content):

    root = ET.fromstring(
        content
    )


    features = []


    for parcel in root.iter():

        if (
            local_name(parcel.tag)
            != "CadastralParcel"
        ):

            continue


        properties = {}


        for child in list(parcel):

            name = local_name(
                child.tag
            )


            if name in PARCEL_FIELDS:

                properties[name] = (
                    child.text or ""
                ).strip()


        rings = []


        for element in parcel.iter():

            if (
                local_name(element.tag)
                == "posList"
                and element.text
            ):

                ring = pairs_from_poslist(
                    element.text
                )


                if ring:

                    rings.append(
                        ring
                    )


        if not rings:

            continue


        feature = {

            "type":
                "Feature",

            "properties":
                properties,

            "geometry": {

                "type":
                    "Polygon",

                "coordinates":
                    rings

            }

        }


        features.append(
            feature
        )


    return {

        "type":
            "FeatureCollection",

        "numberMatched":
            root.attrib.get(
                "numberMatched"
            ),

        "numberReturned":
            root.attrib.get(
                "numberReturned"
            ),

        "features":
            features

    }


# =========================================================
# WFS REQUEST
# =========================================================

def fetch_wfs_bbox(bbox):

    params = {

        "SERVICE":
            "WFS",

        "REQUEST":
            "GetFeature",

        "VERSION":
            "2.0.0",

        "TYPENAMES":
            PARCEL_LAYER,

        "SRSNAME":
            GML_SRS,

        "BBOX":
            bbox,

        "COUNT":
            "200"

    }


    response = SESSION.get(

        WFS_URL,

        params=params,

        headers=HEADERS,

        timeout=TIMEOUT

    )


    response.raise_for_status()


    return extract_parcels(
        response.content
    )


# =========================================================
# FIND PARCEL CONTAINING A POINT
# =========================================================

def find_parcel_at_point(
    latitude,
    longitude,
    radius=0.00025
):

    from shapely.geometry import Point, shape


    south = latitude - radius
    north = latitude + radius

    west = longitude - radius
    east = longitude + radius


    bbox = (
        f"{south},"
        f"{west},"
        f"{north},"
        f"{east}"
    )


    data = fetch_wfs_bbox(
        bbox
    )


    point = Point(
        longitude,
        latitude
    )


    for feature in data.get(
        "features",
        []
    ):

        geometry = feature.get(
            "geometry"
        )


        if not geometry:
            continue


        try:

            polygon = shape(
                geometry
            )


        except Exception as e:

            print(
                "PARCEL GEOMETRY ERROR:",
                e
            )

            continue


        if (
            polygon.contains(point)
            or
            polygon.touches(point)
        ):

            return feature


    return None

# =========================================================
# BUILD WFS TILES
# =========================================================

def build_wfs_tiles(
    south,
    west,
    north,
    east
):

    tiles = []


    lat = south


    while lat < north:

        tile_north = min(
            lat + WFS_TILE_SIZE,
            north
        )


        lon = west


        while lon < east:

            tile_east = min(
                lon + WFS_TILE_SIZE,
                east
            )


            tiles.append(
                f"{lat:.6f},"
                f"{lon:.6f},"
                f"{tile_north:.6f},"
                f"{tile_east:.6f}"
            )


            lon = tile_east


        lat = tile_north


    return tiles


# =========================================================
# FEATURE KEY
# =========================================================

def feature_key(feature):

    properties = feature.get(
        "properties",
        {}
    )


    reference = (

        properties.get(
            "NATIONALCADASTRALREFERENCE",
            ""
        )

        or

        properties.get(
            "INSPIREID_LOCALID",
            ""
        )

    )


    geometry = feature.get(
        "geometry",
        {}
    )


    return (

        normalize_value(
            reference
        ),

        str(
            geometry.get(
                "coordinates",
                ""
            )
        )

    )


# =========================================================
# FETCH TILED WFS
# =========================================================

def fetch_wfs_tiled(bbox):

    cached = _cache_get(
        _foglio_parcels_cache,
        bbox
    )

    if cached is not None:
        return cached

    south, west, north, east = [
        float(value)
        for value in bbox.split(",")
    ]


    tiles = build_wfs_tiles(

        south,
        west,
        north,
        east

    )


    if len(tiles) > WFS_MAX_TILES:

        raise ValueError(

            "The requested map area is too large "
            "for cadastral search. Please zoom in "
            "and try again."

        )


    all_features = []


    number_matched = None


    number_returned = 0


    # ---------------------------------------------------------
    # Tiles are independent WFS requests, so fetch them
    # concurrently instead of one at a time.
    # ---------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=WFS_MAX_WORKERS
    ) as executor:

        future_to_tile = {
            executor.submit(fetch_wfs_bbox, tile_bbox): tile_bbox
            for tile_bbox in tiles
        }

        for future in as_completed(future_to_tile):

            data = future.result()

            if number_matched is None:

                number_matched = data.get(
                    "numberMatched"
                )


            number_returned += len(
                data.get(
                    "features",
                    []
                )
            )


            all_features.extend(
                data.get(
                    "features",
                    []
                )
            )


    # -----------------------------------------------------
    # Remove duplicates caused by neighboring tiles.
    # -----------------------------------------------------

    unique_features = []


    seen = set()


    for feature in all_features:

        key = feature_key(
            feature
        )


        if key in seen:
            continue


        seen.add(key)


        unique_features.append(
            feature
        )


    result = {

        "type":
            "FeatureCollection",

        "numberMatched":
            number_matched,

        "numberReturned":
            number_returned,

        "features":
            unique_features,

        "tile_count":
            len(tiles)

    }

    _cache_set(
        _foglio_parcels_cache,
        bbox,
        result
    )

    return result


# =========================================================
# FETCH EXACT PARCEL FROM F O G L I O BBOX
# =========================================================

def fetch_exact_parcel(
    bbox,
    reference_variants
):

    south, west, north, east = [
        float(value)
        for value in bbox.split(",")
    ]

    GRID_SIZE = 0.002

    tile_bboxes = []

    lat = south

    while lat < north:

        tile_north = min(
            lat + GRID_SIZE,
            north
        )

        lon = west

        while lon < east:

            tile_east = min(
                lon + GRID_SIZE,
                east
            )

            tile_bboxes.append(
                f"{lat:.8f},"
                f"{lon:.8f},"
                f"{tile_north:.8f},"
                f"{tile_east:.8f}"
            )

            lon = tile_east

        lat = tile_north

    found_feature = None

    executor = ThreadPoolExecutor(
        max_workers=WFS_MAX_WORKERS
    )

    try:

        future_to_tile = {
            executor.submit(fetch_wfs_bbox, tile_bbox): tile_bbox
            for tile_bbox in tile_bboxes
        }

        for future in as_completed(future_to_tile):

            data = future.result()

            for feature in data.get(
                "features",
                []
            ):

                if feature_matches(
                    feature,
                    reference_variants
                ):

                    found_feature = feature

                    break

            if found_feature is not None:
                break

    finally:

        executor.shutdown(
            wait=False,
            cancel_futures=True
        )

    return found_feature


# =========================================================
# STRICT CADASTRAL REFERENCE MATCH
# =========================================================

def feature_matches(
    feature,
    reference_variants
):

    properties = (
        feature.get(
            "properties",
            {}
        )
    )


    wanted = {

        normalize_value(
            value
        )

        for value
        in reference_variants

        if value

    }


    cadastral_reference = (
        normalize_value(
            properties.get(
                "NATIONALCADASTRALREFERENCE",
                ""
            )
        )
    )


    local_id = (
        normalize_value(
            properties.get(
                "INSPIREID_LOCALID",
                ""
            )
        )
    )


    if (
        cadastral_reference
        and
        cadastral_reference in wanted
    ):

        return True


    if (
        local_id
        and
        local_id in wanted
    ):

        return True


    return False


# =========================================================
# FIND MATCHED REFERENCE
# =========================================================

def get_matched_reference(
    feature,
    reference_variants
):

    properties = (
        feature.get(
            "properties",
            {}
        )
    )


    wanted = {

        normalize_value(
            value
        )

        for value
        in reference_variants

        if value

    }


    for field in [
        "NATIONALCADASTRALREFERENCE",
        "INSPIREID_LOCALID"
    ]:

        value = normalize_value(
            properties.get(
                field,
                ""
            )
        )


        if value in wanted:

            return value


    return None


# =========================================================
# ADD TARGET INFORMATION
# =========================================================

def mark_target_features(
    features,
    reference_variants
):

    result = []


    for feature in features:

        feature_copy = dict(
            feature
        )


        properties = dict(
            feature.get(
                "properties",
                {}
            )
        )


        matched_reference = (
            get_matched_reference(
                feature,
                reference_variants
            )
        )


        properties["IS_TARGET"] = (
            matched_reference is not None
        )


        if matched_reference:

            properties[
                "MATCHED_CADASTRAL_REFERENCE"
            ] = matched_reference


        feature_copy[
            "properties"
        ] = properties


        result.append(
            feature_copy
        )


    return result


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# COMUNI
# =========================================================

@app.route("/comuni")
def comuni():

    items = []


    for name, code in sorted(
        COMUNI.items()
    ):

        items.append({

            "name":
                name.title(),

            "code":
                code,

            "bbox":
                COMUNE_BBOXES.get(
                    code
                )

        })


    return jsonify({

        "source":
            "Configured local COMUNI list. "
            "The Agenzia Entrate WFS does not "
            "publish a dedicated all-comuni "
            "dropdown catalogue.",

        "count":
            len(items),

        "comuni":
            items

    })

def fetch_cadastral_zoning(bbox):
    params = {
        "SERVICE": "WFS",
        "REQUEST": "GetFeature",
        "VERSION": "2.0.0",
        "TYPENAMES": "CP:CadastralZoning",
        "SRSNAME": GML_SRS,
        "BBOX": bbox,
        "COUNT": "200"
    }

    response = SESSION.get(
        WFS_URL,
        params=params,
        headers=HEADERS,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    return extract_zonings(response.content)


def extract_zonings(content):
    root = ET.fromstring(content)

    features = []

    for zoning in root.iter():

        if local_name(zoning.tag) != "CadastralZoning":
            continue

        properties = {}

        for child in list(zoning):
            name = local_name(child.tag)

            if name in {
                "INSPIREID_LOCALID",
                "INSPIREID_NAMESPACE",
                "LABEL",
                "NATIONALCADASTRALZONINGREFERENCE",
                "ADMINISTRATIVEUNIT",
                "LEVEL",
                "LEVELNAME"
            }:
                properties[name] = (
                    child.text or ""
                ).strip()

        rings = []

        for element in zoning.iter():

            if (
                local_name(element.tag) == "posList"
                and element.text
            ):
                ring = pairs_from_poslist(
                    element.text
                )

                if ring:
                    rings.append(ring)

        if not rings:
            continue

        features.append({
            "type": "Feature",
            "properties": properties,
            "geometry": {
                "type": "Polygon",
                "coordinates": rings
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }

def find_foglio_zoning(bbox, comune_code, foglio_number):

    data = fetch_cadastral_zoning(bbox)

    wanted = (
        f"{comune_code}_"
        f"{foglio_number:04d}00"
    )

    for feature in data.get("features", []):

        props = feature.get(
            "properties",
            {}
        )

        reference = normalize_value(
            props.get(
                "NATIONALCADASTRALZONINGREFERENCE",
                ""
            )
        )

        administrative_unit = normalize_value(
            props.get(
                "ADMINISTRATIVEUNIT",
                ""
            )
        )

        if (
            administrative_unit ==
            normalize_value(comune_code)
            and
            reference == wanted
        ):
            return feature

    return None

def find_foglio_zoning_tiled(
    comune_code,
    foglio_number,
    exact_reference=None
):
    """
    Find a cadastral Foglio polygon by searching
    the configured Comune BBOX in small tiles.
    """

    foglio_digits_for_cache = f"{int(foglio_number):04d}"

    cache_key = (
        comune_code,
        foglio_digits_for_cache,
        normalize_value(exact_reference) if exact_reference else ""
    )

    cached = _cache_get(
        _foglio_zoning_cache,
        cache_key
    )

    if cached is not None:
        return cached

    comune_bbox = COMUNE_BBOXES.get(
        comune_code
    )

    if not comune_bbox:
        return None

    south, west, north, east = [
        float(value)
        for value in comune_bbox.split(",")
    ]

    # Larger tiles are appropriate for locating the Foglio
    # polygon. The WFS silently returns 0 features above
    # roughly a 0.07-0.075 degree bbox edge regardless of
    # feature density (confirmed empirically, including in
    # the densest part of the Comune) - 0.06 stays safely
    # under that with a comfortable margin, while cutting
    # the number of tiles (and requests) needed to scan a
    # whole Comune by roughly 9x compared to 0.02.
    TILE_SIZE = 0.06

    tiles = []

    lat = south

    while lat < north:

        tile_north = min(
            lat + TILE_SIZE,
            north
        )

        lon = west

        while lon < east:

            tile_east = min(
                lon + TILE_SIZE,
                east
            )

            tiles.append(
                f"{lat:.6f},"
                f"{lon:.6f},"
                f"{tile_north:.6f},"
                f"{tile_east:.6f}"
            )

            lon = tile_east

        lat = tile_north

    # Cadastral zoning references can use different
    # two-character suffixes for the same Foglio.
    #
    # Examples:
    # G273_003300
    # G273_0033A0
    # G273_0033B0
    #
    # Accept all known forms.

    foglio_digits = (
        f"{int(foglio_number):04d}"
    )

    target_references = {
        f"{comune_code}_{foglio_digits}00",
        f"{comune_code}_{foglio_digits}A0",
        f"{comune_code}_{foglio_digits}B0",
    }

    print(
        "SEARCHING FOR FOGLIO:",
        sorted(target_references)
    )

    normalized_target_references = {
        normalize_value(value)
        for value in target_references
    }

    normalized_exact_reference = (
        normalize_value(exact_reference)
        if exact_reference
        else None
    )

    normalized_comune_code = normalize_value(
        comune_code
    )

    def tile_matches(data):

        matches = []

        for feature in data.get(
            "features",
            []
        ):

            props = feature.get(
                "properties",
                {}
            )

            reference = normalize_value(
                props.get(
                    "NATIONALCADASTRALZONINGREFERENCE",
                    ""
                )
            )

            administrative_unit = normalize_value(
                props.get(
                    "ADMINISTRATIVEUNIT",
                    ""
                )
            )

            if (
                administrative_unit ==
                normalized_comune_code
                and
                (
                    reference ==
                    normalized_exact_reference
                    if normalized_exact_reference
                    else
                    reference in normalized_target_references
                )
            ):

                matches.append(
                    (feature, reference)
                )

        return matches

    # ---------------------------------------------------------
    # Tiles are independent WFS requests, so scan them
    # concurrently instead of one at a time.
    #
    # A Foglio can be split across several "sviluppo"
    # sections (suffixes 00 / A0 / B0), each covering a
    # different, disjoint part of the Foglio. We can't stop
    # at the first section found - the requested Particella
    # may only exist in a different section - so when no
    # exact_reference was given we keep scanning every tile
    # and collect every matching section. Only an exact,
    # unambiguous reference lookup can stop early.
    # ---------------------------------------------------------

    found_features = {}

    executor = ThreadPoolExecutor(
        max_workers=WFS_MAX_WORKERS
    )

    try:

        future_to_tile = {
            executor.submit(fetch_cadastral_zoning, tile_bbox): tile_bbox
            for tile_bbox in tiles
        }

        for future in as_completed(future_to_tile):

            try:
                data = future.result()

            except requests.RequestException as e:

                print(
                    "ZONING TILE ERROR:",
                    e
                )

                continue

            for feature, reference in tile_matches(data):

                if reference not in found_features:

                    print(
                        "FOGLIO SECTION FOUND:",
                        reference
                    )

                    found_features[reference] = feature

            if (
                normalized_exact_reference
                and found_features
            ):
                break

    finally:

        executor.shutdown(
            wait=False,
            cancel_futures=True
        )

    result = list(
        found_features.values()
    )

    if result:

        _cache_set(
            _foglio_zoning_cache,
            cache_key,
            result
        )

        return result

    print(
        "FOGLIO NOT FOUND:",
        sorted(target_references)
    )

    return []
# =========================================================
# BUILDING FOOTPRINTS
# =========================================================
#
# Query the Regione Siciliana Fabbricati layer and return
# building polygons intersecting the target Particella.
#
# This is intentionally separate from the existing parcel
# search so the cadastral matching logic remains unchanged.
# =========================================================

FABBRICATI_URL = (
    "https://map.sitr.regione.sicilia.it/"
    "gis/rest/services/catasto/cartografia_catastale/"
    "MapServer/7/query"
)


def fetch_building_footprints(parcel_feature):

    geometry = (
        parcel_feature.get("geometry")
        if parcel_feature
        else None
    )

    if not geometry:
        print(
            "BUILDING FOOTPRINTS: target has no geometry."
        )
        return []

    try:

        parcel = shape(
            geometry
        )

    except Exception as e:

        print(
            "BUILDING PARCEL GEOMETRY ERROR:",
            e
        )

        return []

    if parcel.is_empty:
        print(
            "BUILDING FOOTPRINTS: target geometry is empty."
        )
        return []

    minx, miny, maxx, maxy = parcel.bounds

    print()
    print("BUILDING QUERY BBOX:")
    print("west :", minx)
    print("south:", miny)
    print("east :", maxx)
    print("north:", maxy)

    params = {

        "where":
            "1=1",

        "geometry":
            f"{minx},{miny},{maxx},{maxy}",

        "geometryType":
            "esriGeometryEnvelope",

        "inSR":
            "4326",

        "spatialRel":
            "esriSpatialRelIntersects",

        "outFields":
            "*",

        "returnGeometry":
            "true",

        "outSR":
            "4326",

        "f":
            "geojson"
    }

    try:

        response = requests.get(
            FABBRICATI_URL,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        result = response.json()

    except Exception as e:

        print(
            "BUILDING FOOTPRINT QUERY ERROR:",
            e
        )

        return []

    if "error" in result:

        print(
            "BUILDING FOOTPRINT SERVICE ERROR:",
            result.get("error")
        )

        return []

    returned = result.get(
        "features",
        []
    )

    print(
        "FABBRICATI FEATURES RETURNED:",
        len(returned)
    )

    building_features = []

    for feature in returned:

        building_geometry = (
            feature.get(
                "geometry"
            )
        )

        if not building_geometry:
            continue

        try:

            building = shape(
                building_geometry
            )

        except Exception as e:

            print(
                "BUILDING GEOMETRY ERROR:",
                e
            )

            continue

        if building.is_empty:
            continue

        if not parcel.intersects(
            building
        ):
            continue

        building_features.append(
            feature
        )

        props = feature.get(
            "properties",
            {}
        )

        print(
            "BUILDING INTERSECTION:",
            "OBJECTID=",
            props.get("OBJECTID"),
            "FOGLIO=",
            props.get("FOGLIO"),
            "NUMERO=",
            props.get("NUMERO")
        )

    print(
        "BUILDING FOOTPRINTS FOUND:",
        len(building_features)
    )

    return building_features


# =========================================================
# COORDINATE SEARCH
# =========================================================
#
# Convert geographical coordinates into the cadastral
# Foglio + Particella containing that point.
#
# This route only identifies the parcel.
# The existing /search route remains responsible for
# returning the normal cadastral polygon and target styling.
# =========================================================

@app.route("/search-coordinate")
def search_coordinate():

    lat_text = request.args.get(
        "lat",
        ""
    ).strip()

    lon_text = request.args.get(
        "lon",
        ""
    ).strip()

    if not lat_text or not lon_text:

        return jsonify({
            "error":
                "Latitude and longitude are required."
        }), 400

    try:

        latitude = float(lat_text)
        longitude = float(lon_text)

    except ValueError:

        return jsonify({
            "error":
                "Latitude and longitude must be valid numbers."
        }), 400

    if not (-90 <= latitude <= 90):

        return jsonify({
            "error":
                "Latitude must be between -90 and 90."
        }), 400

    if not (-180 <= longitude <= 180):

        return jsonify({
            "error":
                "Longitude must be between -180 and 180."
        }), 400

    building_url = (
        "https://map.sitr.regione.sicilia.it/"
        "gis/rest/services/catasto/cartografia_catastale/"
        "MapServer/4/query"
    )

    params = {

        "f":
            "json",

        "where":
            "1=1",

        "geometry":
            f"{longitude},{latitude}",

        "geometryType":
            "esriGeometryPoint",

        "inSR":
            "4326",

        "spatialRel":
            "esriSpatialRelIntersects",

        "outFields":
            "*",

        "returnGeometry":
            "true"
    }

    try:

        response = requests.get(
            building_url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:

        return jsonify({
            "error":
                f"Cadastral coordinate search failed: {e}"
        }), 502

    except ValueError:

        return jsonify({
            "error":
                "Cadastral service returned invalid JSON."
        }), 502

    features = data.get(
        "features",
        []
    )

    if not features:

        return jsonify({

            "error":
                "No cadastral parcel was found at these coordinates.",

            "coordinates": {

                "lat":
                    latitude,

                "lon":
                    longitude
            }

        }), 404

    feature = features[0]

    properties = feature.get(
        "attributes",
        {}
    )

    foglio = str(
        properties.get(
            "FOGLIO",
            ""
        )
    ).strip()

    particella = str(
        properties.get(
            "NUMERO",
            ""
        )
    ).strip()

    comune_code = str(
        properties.get(
            "COMUNE",
            ""
        )
    ).strip()

    if not foglio or not particella:

        return jsonify({

            "error":
                "Cadastral parcel was found, but Foglio/Particella could not be determined.",

            "attributes":
                properties
        }), 500

    print()
    print("==============================")
    print("COORDINATE CADASTRAL SEARCH")
    print("==============================")
    print("Latitude:", latitude)
    print("Longitude:", longitude)
    print("Comune:", comune_code)
    print("Foglio:", foglio)
    print("Particella:", particella)

    return jsonify({

        "found":
            True,

        "coordinates": {

            "lat":
                latitude,

            "lon":
                longitude
        },

        "comune_code":
            comune_code,

        "foglio":
            foglio,

        "particella":
            particella,

        "properties":
            properties,

        "geometry":
            feature.get(
                "geometry"
            )
    })


# =========================================================
# SEARCH
# =========================================================

@app.route("/search")
def search():

    comune = request.args.get(
        "comune",
        ""
    ).strip()

    foglio = request.args.get(
        "foglio",
        ""
    ).strip()

    particella = request.args.get(
        "particella",
        ""
    ).strip()

    requested_bbox = request.args.get(
        "bbox",
        ""
    ).strip()


    # =====================================================
    # VALIDATE COMUNE
    # =====================================================

    comune_code = get_comune_code(
        comune
    )

    if not comune_code:

        return jsonify({
            "error":
                f"Comune '{comune}' "
                "is not configured."
        }), 400


    # =====================================================
    # VALIDATE FOGLIO
    # =====================================================

    if not foglio:

        return jsonify({
            "error":
                "Foglio is required."
        }), 400

    try:

        foglio_number = int(
            foglio
        )

        if foglio_number < 0:
            raise ValueError

    except ValueError:

        return jsonify({
            "error":
                "Foglio must be a valid number."
        }), 400


    # =====================================================
    # VALIDATE PARTICELLA
    # =====================================================

    if not particella:

        return jsonify({
            "error":
                "Particella is required."
        }), 400


    normalized_particella = (
        normalize_number(
            particella
        )
    )


    if not normalized_particella:

        return jsonify({
            "error":
                "Particella must be a valid number."
        }), 400


    # =====================================================
    # BUILD EXPECTED REFERENCES
    # =====================================================

    reference_variants = (
        build_reference_variants(
            comune_code,
            str(foglio_number),
            normalized_particella
        )
    )


    print()
    print("==============================")
    print("AUTOMATIC CADASTRAL SEARCH")
    print("==============================")
    print("Comune:", comune)
    print("Comune code:", comune_code)
    print("Foglio:", foglio_number)
    print("Particella:", normalized_particella)


    # =====================================================
    # FIND THE F O G L I O
    #
    # IMPORTANT:
    # We no longer depend on the visible map BBOX.
    # =====================================================

    try:

        foglio_zoning_sections = (
            find_foglio_zoning_tiled(
                comune_code,
                foglio_number
            )
        )

    except Exception as e:

        print(
            "FOGLIO SEARCH ERROR:",
            e
        )

        return jsonify({
            "error":
                f"Unable to locate Foglio "
                f"{foglio_number}: {e}"
        }), 500


    if not foglio_zoning_sections:

        return jsonify({

            "error":
                f"Foglio {foglio_number} "
                f"was not found in "
                f"{comune}.",

            "searched": {
                "comune": comune,
                "comune_code": comune_code,
                "foglio": str(foglio_number),
                "particella":
                    normalized_particella
            }

        }), 404


    # =====================================================
    # CALCULATE F O G L I O BBOX
    #
    # A Foglio may be split across multiple "sviluppo"
    # sections (00 / A0 / B0, ...), each covering a
    # different part of it. We union the bounds of every
    # section found so the Particella is covered no matter
    # which section it actually lives in.
    # =====================================================

    min_lat = None
    min_lon = None
    max_lat = None
    max_lon = None


    for foglio_zoning in foglio_zoning_sections:

        geometry = foglio_zoning.get(
            "geometry",
            {}
        )

        coordinates = geometry.get(
            "coordinates",
            []
        )


        for ring in coordinates:

            for point in ring:

                if len(point) < 2:
                    continue

                lon = float(
                    point[0]
                )

                lat = float(
                    point[1]
                )


                min_lat = (
                    lat
                    if min_lat is None
                    else min(
                        min_lat,
                        lat
                    )
                )

                max_lat = (
                    lat
                    if max_lat is None
                    else max(
                        max_lat,
                        lat
                    )
                )

                min_lon = (
                    lon
                    if min_lon is None
                    else min(
                        min_lon,
                        lon
                    )
                )

                max_lon = (
                    lon
                    if max_lon is None
                    else max(
                        max_lon,
                        lon
                    )
                )


    if (
        min_lat is None
        or min_lon is None
        or max_lat is None
        or max_lon is None
    ):

        return jsonify({
            "error":
                "Foglio polygon has no usable geometry."
        }), 500


    # Small padding around Foglio.
    padding = 0.0002

    parcel_bbox = (
        f"{min_lat - padding},"
        f"{min_lon - padding},"
        f"{max_lat + padding},"
        f"{max_lon + padding}"
    )


    print(
        "FOGLIO BBOX:",
        parcel_bbox
    )


    # =====================================================
    # FETCH PARCELS INSIDE THE F O G L I O
    # =====================================================

    try:

        data = fetch_wfs_tiled(
            parcel_bbox
        )

    except requests.RequestException as e:

        return jsonify({
            "error":
                f"WFS connection error: {e}"
        }), 500

    except (
        ET.ParseError,
        ValueError
    ) as e:

        return jsonify({
            "error":
                f"WFS parse/search error: {e}"
        }), 500


    returned_features = data.get(
        "features",
        []
    )


    # =====================================================
    # FALLBACK EXACT PARTICELLA SEARCH
    #
    # The tiled fetch above already covers the whole Foglio
    # bbox, so it normally already contains the target
    # Particella. Only fall back to the slower fine-grained
    # grid scan when it's genuinely missing.
    # =====================================================

    already_found = any(
        feature_matches(
            feature,
            reference_variants
        )
        for feature in returned_features
    )

    exact_feature = (
        None
        if already_found
        else fetch_exact_parcel(
            parcel_bbox,
            reference_variants
        )
    )

    if exact_feature:

        exact_key = feature_key(
            exact_feature
        )

        if not any(
            feature_key(feature) == exact_key
            for feature in returned_features
        ):

            returned_features.append(
                exact_feature
            )

        print(
            "EXACT PARTICELLA FOUND:",
            exact_feature.get(
                "properties",
                {}
            ).get(
                "NATIONALCADASTRALREFERENCE"
            )
        )


    print(
        "PARCELS RETURNED:",
        len(returned_features)
    )


    # =====================================================
    # FIND EXACT PARTICELLA
    # =====================================================

    matched = [

        feature

        for feature
        in returned_features

        if feature_matches(
            feature,
            reference_variants
        )

    ]


    matched_reference = None


    if matched:

        matched_reference = (
            get_matched_reference(
                matched[0],
                reference_variants
            )
        )


    print(
        "MATCHED:",
        bool(matched)
    )

    print(
        "MATCHED REFERENCE:",
        matched_reference
    )


    # =====================================================
    # MARK ONLY EXACT TARGET
    # =====================================================

    marked_features = (
        mark_target_features(
            returned_features,
            reference_variants
        )
    )


    # =====================================================
    # BUILDING FOOTPRINTS FOR TARGET PARTICELLA
    # =====================================================

    target_feature = None

    for feature in marked_features:

        props = (
            feature.get(
                "properties",
                {}
            )
        )

        is_target = (
            props.get("IS_TARGET") is True
            or
            props.get("IS_TARGET") == "true"
            or
            props.get("IS_TARGET") == 1
            or
            props.get("IS_TARGET") == "1"
        )

        if is_target:

            target_feature = feature
            break


    building_footprints = (
        fetch_building_footprints(
            target_feature
        )
        if target_feature
        else []
    )


    # =====================================================
    # RESPONSE
    # =====================================================

    result = {

        "type":
            "FeatureCollection",

        "features":
            marked_features,

        "building_footprints":
            building_footprints,

        "tile_count":
            data.get(
                "tile_count",
                0
            ),

        "numberMatched":
            data.get(
                "numberMatched"
            ),

        "numberReturned":
            data.get(
                "numberReturned"
            ),

        "searched": {

            "comune":
                comune,

            "comune_code":
                comune_code,

            "foglio":
                str(
                    foglio_number
                ),

            "particella":
                normalized_particella,

            "references":
                sorted(
                    reference_variants
                ),

            "matched":
                bool(
                    matched
                ),

            "matched_count":
                len(
                    matched
                ),

            "matched_reference":
                matched_reference,

            "returned_features":
                len(
                    returned_features
                ),

            "bbox":
                parcel_bbox,

            "bbox_source":
                "foglio",

            "requested_map_bbox":
                requested_bbox,

            "tile_count":
                data.get(
                    "tile_count",
                    0
                )
        },

        "foglio_zoning_found":
            True,

        "foglio_zoning_sections":
            foglio_zoning_sections
    }


    # =====================================================
    # WARNING
    # =====================================================

    if matched:

        result["warning"] = None

    elif not returned_features:

        result["warning"] = (
            "No cadastral parcels were "
            "returned inside the Foglio."
        )

    else:

        result["warning"] = (
            "The Foglio was found, but the "
            "requested Particella was not "
            "found inside the returned parcels."
        )


    print("==============================")
    print()


    return jsonify(
        result
    )
# =========================================================
# DEBUG SEARCH
# =========================================================

@app.route("/debug-search")
def debug_search():

    return jsonify({

        "comune":
            request.args.get(
                "comune",
                ""
            ),

        "foglio":
            request.args.get(
                "foglio",
                ""
            ),

        "particella":
            request.args.get(
                "particella",
                ""
            ),

        "bbox":
            request.args.get(
                "bbox",
                ""
            )

    })


# =========================================================
# FIND F O G L I O
# =========================================================

@app.route("/search-foglio")
def search_foglio():

    comune = request.args.get(
        "comune",
        ""
    ).strip()

    foglio = request.args.get(
        "foglio",
        ""
    ).strip()

    # -----------------------------------------------------
    # Validate Comune
    # -----------------------------------------------------

    comune_code = get_comune_code(
        comune
    )

    if not comune_code:

        return jsonify({
            "error":
                f"Comune '{comune}' is not configured."
        }), 400

    # -----------------------------------------------------
    # Validate Foglio
    # -----------------------------------------------------

    if not foglio:

        return jsonify({
            "error":
                "Foglio is required."
        }), 400

    try:

        foglio_number = int(
            foglio
        )

        if foglio_number < 0:
            raise ValueError

    except ValueError:

        return jsonify({
            "error":
                "Foglio must be a valid number."
        }), 400

    # -----------------------------------------------------
    # Get Comune BBOX
    # -----------------------------------------------------

    comune_bbox = COMUNE_BBOXES.get(
        comune_code
    )

    if not comune_bbox:

        return jsonify({
            "error":
                f"No BBOX configured for {comune_code}."
        }), 400

    south, west, north, east = [
        float(value)
        for value in comune_bbox.split(",")
    ]

    # -----------------------------------------------------
    # We search using larger tiles here.
    #
    # This is ONLY for locating the Foglio.
    # Once found, the exact parcel search can use
    # the smaller BBOX.
    # -----------------------------------------------------

    SEARCH_TILE_SIZE = 0.02

    tiles = []

    lat = south

    while lat < north:

        tile_north = min(
            lat + SEARCH_TILE_SIZE,
            north
        )

        lon = west

        while lon < east:

            tile_east = min(
                lon + SEARCH_TILE_SIZE,
                east
            )

            tiles.append(
                f"{lat:.6f},"
                f"{lon:.6f},"
                f"{tile_north:.6f},"
                f"{tile_east:.6f}"
            )

            lon = tile_east

        lat = tile_north

    # -----------------------------------------------------
    # Safety limit
    # -----------------------------------------------------

    MAX_FOGLIO_TILES = 250

    if len(tiles) > MAX_FOGLIO_TILES:

        return jsonify({
            "error":
                "Comune search area is too large.",
            "tile_count":
                len(tiles)
        }), 400


    # -----------------------------------------------------
    # Search
    # -----------------------------------------------------

    matching_features = []

    searched_tiles = 0

    target_foglio = f"{foglio_number:04d}00"

    for tile_bbox in tiles:

        searched_tiles += 1

        try:

            data = fetch_wfs_bbox(
                tile_bbox
            )

        except requests.RequestException:

            continue

        for feature in data.get(
            "features",
            []
        ):

            props = feature.get(
                "properties",
                {}
            )

            administrative_unit = normalize_value(
                props.get(
                    "ADMINISTRATIVEUNIT",
                    ""
                )
            )

            national = normalize_value(
                props.get(
                    "NATIONALCADASTRALREFERENCE",
                    ""
                )
            )

            local_id = normalize_value(
                props.get(
                    "INSPIREID_LOCALID",
                    ""
                )
            )

            # -------------------------------------------------
            # Comune check
            # -------------------------------------------------

            belongs_to_comune = (

                administrative_unit ==
                normalize_value(
                    comune_code
                )

                or

                national.startswith(
                    comune_code + "_"
                )

                or

                local_id.startswith(
                    "IT.AGE.PLA."
                    + comune_code
                    + "_"
                )

            )

            if not belongs_to_comune:

                continue

            # -------------------------------------------------
            # Foglio check
            # -------------------------------------------------

            reference_matches_foglio = (

                f"{comune_code}_{target_foglio}"
                in national

                or

                f"{comune_code}_{target_foglio}"
                in local_id

            )

            if reference_matches_foglio:

                matching_features.append(
                    feature
                )

    # -----------------------------------------------------
    # Remove duplicates
    # -----------------------------------------------------

    unique_features = []

    seen = set()

    for feature in matching_features:

        key = feature_key(
            feature
        )

        if key in seen:
            continue

        seen.add(key)

        unique_features.append(
            feature
        )

    matching_features = unique_features

    # -----------------------------------------------------
    # Calculate BBOX of the Foglio parcels
    # -----------------------------------------------------

    min_lat = None
    min_lon = None
    max_lat = None
    max_lon = None

    for feature in matching_features:

        geometry = feature.get(
            "geometry",
            {}
        )

        coordinates = geometry.get(
            "coordinates",
            []
        )

        for ring in coordinates:

            for point in ring:

                if len(point) < 2:
                    continue

                lon = float(
                    point[0]
                )

                lat = float(
                    point[1]
                )

                min_lat = (
                    lat
                    if min_lat is None
                    else min(min_lat, lat)
                )

                max_lat = (
                    lat
                    if max_lat is None
                    else max(max_lat, lat)
                )

                min_lon = (
                    lon
                    if min_lon is None
                    else min(min_lon, lon)
                )

                max_lon = (
                    lon
                    if max_lon is None
                    else max(max_lon, lon)
                )

    # -----------------------------------------------------
    # Result
    # -----------------------------------------------------

    result = {

        "comune":
            comune,

        "comune_code":
            comune_code,

        "foglio":
            str(foglio_number),

        "target_foglio":
            target_foglio,

        "searched_tiles":
            searched_tiles,

        "matching_parcels":
            len(
                matching_features
            ),

        "features":
            matching_features

    }

    if (
        min_lat is not None
        and
        min_lon is not None
        and
        max_lat is not None
        and
        max_lon is not None
    ):

        result["bbox"] = (
            f"{min_lat},"
            f"{min_lon},"
            f"{max_lat},"
            f"{max_lon}"
        )

    else:

        result["bbox"] = None

    return jsonify(
        result
    )

# =========================================================
# ADDRESS SEARCH
# =========================================================

@app.route("/address-search")
def address_search():

    address = request.args.get(
        "address",
        ""
    ).strip()


    if not address:

        return jsonify({
            "error":
                "Address is required."
        }), 400


    try:

        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": address,
                "format": "json",
                "limit": 1
            },
            headers={
                "User-Agent":
                    "CatastoMapViewer/1.0"
            },
            timeout=15
        )

        response.raise_for_status()

        results = response.json()


    except requests.RequestException as e:

        print(
            "ADDRESS GEOCODER ERROR:",
            e
        )

        return jsonify({
            "error":
                "Address lookup failed.",
            "details":
                str(e)
        }), 502


    if not results:

        return jsonify({
            "error":
                "Address not found."
        }), 404


    result = results[0]


    try:

        latitude = float(
            result["lat"]
        )

        longitude = float(
            result["lon"]
        )

    except (
        KeyError,
        ValueError,
        TypeError
    ):

        return jsonify({
            "error":
                "Address returned invalid coordinates."
        }), 502


    print(
        "ADDRESS FOUND:",
        address,
        latitude,
        longitude
    )


    # -----------------------------------------------------
    # Find cadastral parcel containing the address point.
    # -----------------------------------------------------

    parcel = None

    try:

        parcel = find_parcel_at_point(
            latitude,
            longitude
        )

    except Exception as e:

        print(
            "ADDRESS PARCEL ERROR:",
            e
        )


    # -----------------------------------------------------
    # Build response.
    # -----------------------------------------------------

    response_data = {

        "query":
            address,

        "display_name":
            result.get(
                "display_name",
                address
            ),

        "lat":
            latitude,

        "lon":
            longitude

    }


    if parcel:

        props = parcel.get(
            "properties",
            {}
        )

        reference = normalize_value(
            props.get(
                "NATIONALCADASTRALREFERENCE",
                ""
            )
        )


        response_data["parcel"] = parcel

        response_data["cadastral_reference"] = (
            reference
        )


        # Example:
        # G273_013800.660
        #
        # -> Foglio 138
        # -> Particella 660

        prefix = (
            "G273_"
        )

        if reference.startswith(
            prefix
        ):

            suffix = reference[
                len(prefix):
            ]

            parts = suffix.split(
                ".",
                1
            )

            if len(parts) == 2:

                foglio_text = parts[0]

                # Cadastral references may encode the
                # Foglio as four digits followed by a
                # two-character suffix.
                #
                # Examples:
                # 013300 -> Foglio 33
                # 0033A0 -> Foglio 33
                # 0033B0 -> Foglio 33
                if (
                    len(foglio_text) == 6
                    and foglio_text[:4].isdigit()
                    and foglio_text[4:] in ("00", "A0", "B0")
                ):
                    foglio_text = foglio_text[:4]

                particella_text = parts[1]

                if (
                    foglio_text.isdigit()
                    and
                    particella_text
                ):

                    response_data["foglio"] = str(
                        int(foglio_text)
                    )

                    response_data["particella"] = (
                        particella_text
                    )

                    # -------------------------------------------------
                    # If the address matched a STRADA feature, it is
                    # a cadastral road and NOT a real Particella.
                    #
                    # In that case we still use its Foglio to retrieve
                    # the normal parcels belonging to that Foglio.
                    # -------------------------------------------------

                    if particella_text.upper().startswith("STRADA"):

                        try:

                            # -------------------------------------------------
                            # The STRADA feature gives us the real Foglio
                            # prefix, for example:
                            #
                            # G273_0033B0.STRADA002
                            #
                            # We search a small area around the address point
                            # and keep only normal numeric parcels belonging
                            # to that exact Foglio prefix.
                            # -------------------------------------------------

                            # -------------------------------------------------
                            # Use the complete Foglio polygon, not just a small
                            # box around the address. The address may fall on a
                            # STRADA feature while the real numeric parcels are
                            # elsewhere inside the same Foglio.
                            # -------------------------------------------------

                            # Use the exact zoning reference from the
                            # address result, e.g. G273_0033A0.
                            #
                            # This prevents Foglio 33 from accidentally
                            # selecting another zoning section such as B0.

                            address_reference = normalize_value(
                                reference
                            )

                            exact_foglio_reference = (
                                address_reference.rsplit(
                                    ".",
                                    1
                                )[0]
                                if "." in address_reference
                                else ""
                            )

                            print(
                                "ADDRESS EXACT ZONING REFERENCE:",
                                exact_foglio_reference
                            )

                            foglio_zoning_sections = find_foglio_zoning_tiled(
                                "G273",
                                int(foglio_text),
                                exact_reference=exact_foglio_reference
                            )

                            if not foglio_zoning_sections:
                                raise ValueError(
                                    f"Foglio {foglio_text} zoning polygon "
                                    "could not be found."
                                )

                            # exact_reference is unambiguous, so there's
                            # only ever one matching section here.
                            foglio_zoning = foglio_zoning_sections[0]

                            geometry = foglio_zoning.get(
                                "geometry",
                                {}
                            )

                            coordinates = geometry.get(
                                "coordinates",
                                []
                            )

                            min_lat = None
                            min_lon = None
                            max_lat = None
                            max_lon = None

                            def collect_points(value):
                                if (
                                    isinstance(value, (list, tuple))
                                    and len(value) >= 2
                                    and isinstance(value[0], (int, float))
                                    and isinstance(value[1], (int, float))
                                ):
                                    yield value
                                    return

                                if isinstance(value, (list, tuple)):
                                    for item in value:
                                        yield from collect_points(item)

                            for point in collect_points(coordinates):

                                lon = float(point[0])
                                lat = float(point[1])

                                min_lat = (
                                    lat
                                    if min_lat is None
                                    else min(min_lat, lat)
                                )

                                max_lat = (
                                    lat
                                    if max_lat is None
                                    else max(max_lat, lat)
                                )

                                min_lon = (
                                    lon
                                    if min_lon is None
                                    else min(min_lon, lon)
                                )

                                max_lon = (
                                    lon
                                    if max_lon is None
                                    else max(max_lon, lon)
                                )

                            if (
                                min_lat is None
                                or min_lon is None
                                or max_lat is None
                                or max_lon is None
                            ):
                                raise ValueError(
                                    "Foglio zoning polygon has no usable geometry."
                                )

                            foglio_bbox = (
                                f"{min_lat},"
                                f"{min_lon},"
                                f"{max_lat},"
                                f"{max_lon}"
                            )

                            print(
                                "ADDRESS F O G L I O BBOX:",
                                foglio_bbox
                            )

                            foglio_data = fetch_wfs_tiled(
                                foglio_bbox
                            )

                            foglio_prefix = normalize_value(
                                foglio_zoning.get("properties", {}).get(
                                    "NATIONALCADASTRALZONINGREFERENCE",
                                    ""
                                )
                            )

                            if not foglio_prefix:
                                raise ValueError(
                                    "Foglio zoning has no cadastral reference."
                                )

                            print(
                                "ADDRESS F O G L I O PARCEL PREFIX:",
                                foglio_prefix
                            )

                            print(
                                "ADDRESS DEBUG ZONING:",
                                exact_foglio_reference
                            )

                            print(
                                "ADDRESS DEBUG FOUND ZONING:",
                                foglio_prefix
                            )

                            print(
                                "ADDRESS DEBUG WFS FEATURES:",
                                len(
                                    foglio_data.get(
                                        "features",
                                        []
                                    )
                                )
                            )

                            foglio_features = []

                            for feature in foglio_data.get(
                                "features",
                                []
                            ):

                                feature_props = feature.get(
                                    "properties",
                                    {}
                                )

                                feature_reference = normalize_value(
                                    feature_props.get(
                                        "NATIONALCADASTRALREFERENCE",
                                        ""
                                    )
                                )

                                if not feature_reference.startswith(
                                    foglio_prefix + "."
                                ):
                                    continue

                                feature_label = normalize_value(
                                    feature_props.get(
                                        "LABEL",
                                        ""
                                    )
                                )

                                # Only real numeric Particelle.
                                if feature_label.isdigit():

                                    foglio_features.append(
                                        feature
                                    )

                            response_data["foglio_features"] = (
                                foglio_features
                            )

                            response_data["foglio_parcels"] = len(
                                foglio_features
                            )

                            # STRADA is not a real Particella.
                            response_data["particella"] = None

                            response_data["address_feature_type"] = (
                                "road"
                            )

                            print(
                                "ADDRESS F O G L I O:",
                                foglio_prefix
                            )

                            print(
                                "ADDRESS F O G L I O PARCELS:",
                                len(foglio_features)
                            )

                        except Exception as e:

                            print(
                                "ADDRESS F O G L I O PARCEL ERROR:",
                                e
                            )

                            response_data["foglio_features"] = []

                            response_data["foglio_parcels"] = 0


    else:

        response_data["parcel"] = None

        response_data["cadastral_reference"] = None

        response_data["foglio"] = None

        response_data["particella"] = None


    return jsonify(
        response_data
    )


# =========================================================
# BUILDING FOOTPRINTS (fabbricati) INSIDE A PARCEL
# =========================================================

def extract_osm_buildings(data):

    features = []

    for element in data.get("elements", []):

        if element.get("type") != "way":
            continue

        geometry = element.get("geometry")

        if not geometry or len(geometry) < 4:
            continue

        ring = [
            [point["lon"], point["lat"]]
            for point in geometry
            if "lon" in point and "lat" in point
        ]

        if len(ring) < 4:
            continue

        if ring[0] != ring[-1]:
            ring.append(ring[0])

        tags = element.get("tags", {})

        features.append({
            "type": "Feature",
            "properties": {
                "BUILDING": tags.get("building", "yes"),
                "OSM_ID": element.get("id")
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring]
            }
        })

    return features


def fetch_osm_buildings(bbox):

    south, west, north, east = bbox

    cache_key = (
        f"{south:.6f},{west:.6f},"
        f"{north:.6f},{east:.6f}"
    )

    cached = _cache_get(
        _buildings_cache,
        cache_key
    )

    if cached is not None:
        return cached

    query = (
        "[out:json][timeout:20];"
        f'way["building"]({south},{west},{north},{east});'
        "out geom;"
    )

    # The public Overpass instance occasionally returns a
    # transient 504/429 under load. One retry clears most
    # of those without making the caller wait too long.
    attempts = 2

    response = None

    for attempt in range(attempts):

        try:

            response = SESSION.post(
                OVERPASS_URL,
                data={"data": query},
                headers=HEADERS,
                timeout=OVERPASS_TIMEOUT
            )

            response.raise_for_status()

            break

        except requests.RequestException as e:

            if attempt + 1 >= attempts:
                raise

            print(
                "OVERPASS RETRY:",
                e
            )

            time.sleep(1)

    features = extract_osm_buildings(
        response.json()
    )

    _cache_set(
        _buildings_cache,
        cache_key,
        features
    )

    return features


@app.route("/parcel-buildings", methods=["POST"])
def parcel_buildings():

    payload = request.get_json(silent=True) or {}

    geometry = payload.get("geometry")

    if not geometry or geometry.get("type") != "Polygon":

        return jsonify({
            "error":
                "A Polygon geometry is required."
        }), 400

    from shapely.geometry import shape, mapping

    try:

        parcel_polygon = shape(
            geometry
        )

    except Exception as e:

        return jsonify({
            "error":
                f"Invalid parcel geometry: {e}"
        }), 400

    min_lon, min_lat, max_lon, max_lat = (
        parcel_polygon.bounds
    )

    # Small padding so buildings whose footprint straddles
    # the parcel boundary are still fetched from Overpass -
    # the shapely intersects() check below still restricts
    # what's returned to the parcel itself.
    padding = 0.0001

    bbox = (
        min_lat - padding,
        min_lon - padding,
        max_lat + padding,
        max_lon + padding
    )

    try:

        candidate_buildings = fetch_osm_buildings(
            bbox
        )

    except requests.RequestException as e:

        return jsonify({
            "error":
                f"Building lookup failed: {e}"
        }), 502

    matching_buildings = []

    for feature in candidate_buildings:

        try:

            building_polygon = shape(
                feature["geometry"]
            )

        except Exception:
            continue

        if not parcel_polygon.intersects(building_polygon):
            continue

        # Show the building's footprint area as a simple
        # box (its minimum rotated bounding rectangle)
        # instead of the real, detailed OSM outline.
        try:

            simplified_polygon = (
                building_polygon.minimum_rotated_rectangle
            )

        except Exception:
            simplified_polygon = building_polygon

        # The box can extend past the parcel edge even when
        # the real building doesn't (or the building itself
        # straddles two parcels) - clip it to the parcel so
        # the shown shape never overlaps a neighbouring
        # Particella.
        try:

            clipped_polygon = parcel_polygon.intersection(
                simplified_polygon
            )

        except Exception:
            continue

        if (
            clipped_polygon.is_empty
            or clipped_polygon.area <= 0
        ):
            continue

        matching_buildings.append({
            "type": "Feature",
            "properties": feature.get(
                "properties",
                {}
            ),
            "geometry": mapping(
                clipped_polygon
            )
        })

    return jsonify({
        "type": "FeatureCollection",
        "features": matching_buildings
    })


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True,

        threaded=True

    )
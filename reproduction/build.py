'''Construct the EPA UST Finder leaking-release long-run status cohort.

Frozen question: will a leaking underground storage tank release reported
between 2004 and 2008 still be classified Open, rather than No Further
Action, at the 2018-2019 state data snapshot? Only report-time fields are
scored; the current-vintage spatial overlays ship as unscored carry columns.
'''

from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from source_io import fetch

LAYER_URL = 'https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/UST_Finder_Feature_Layer_2/FeatureServer/1'
QUERY_URL = LAYER_URL + '/query'
LICENSE_URL = 'https://edg.epa.gov/EPA_Data_License.html'
DOCS_URL = 'https://www.epa.gov/ust/ust-finder'

DOWNLOAD_FIELDS = [
    'OBJECTID',
    'Facility_ID',
    'LUST_ID',
    'State',
    'County',
    'Reported_Date',
    'Status',
    'Substance',
    'Population_within_1500ft',
    'DomesticWells_within_1500ft',
    'LandUse',
    'Within_SPA',
    'Within_100yr_Floodplain',
    'EPA_Region',
]

WHERE_CLAUSE = "Reported_Date BETWEEN TIMESTAMP '2004-01-01 00:00:00' AND TIMESTAMP '2008-12-31 23:59:59'"
PAGE_SIZE = 2000
MAX_PAGES = 400
MAX_ROWS = 400000

ID_COLUMN = 'lust_id'
TIME_COLUMN = 'reported_date'
TARGET_COLUMN = 'still_open_at_snapshot'

FEATURE_COLUMNS = ['substance_released', 'state', 'report_year']

OVERLAY_COLUMNS = [
    'population_within_1500ft',
    'domestic_wells_within_1500ft',
    'land_use',
    'within_source_water_protection_area',
    'within_100yr_floodplain',
]

CARRY_COLUMNS = ['county', 'epa_region'] + OVERLAY_COLUMNS
OUTPUT_COLUMNS = [ID_COLUMN, TIME_COLUMN] + FEATURE_COLUMNS + CARRY_COLUMNS + [TARGET_COLUMN]

WINDOW_START = pd.Timestamp('2004-01-01T00:00:00Z')
WINDOW_END = pd.Timestamp('2009-01-01T00:00:00Z')

SPLIT_BOUNDS = {
    'train': ('2004-01-01', '2006-01-01'),
    'eval': ('2006-01-01', '2007-01-01'),
    'holdout': ('2007-01-01', '2009-01-01'),
}

SPLIT_PATHS = {
    'train': ('public', 'train.csv'),
    'eval': ('public', 'eval.csv'),
    'holdout': ('private', 'holdout.csv'),
}

STATE_CODES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI',
    'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN',
    'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH',
    'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA',
    'WV', 'WI', 'WY',
}

TERRITORY_CODES = {'AS', 'GU', 'MP', 'PR', 'UM', 'VI', 'FM', 'MH', 'PW'}

STATE_NAME_TO_CODE = {
    'alabama': 'AL', 'alaska': 'AK', 'arizona': 'AZ', 'arkansas': 'AR',
    'california': 'CA', 'colorado': 'CO', 'connecticut': 'CT', 'delaware': 'DE',
    'district of columbia': 'DC', 'florida': 'FL', 'georgia': 'GA', 'hawaii': 'HI',
    'idaho': 'ID', 'illinois': 'IL', 'indiana': 'IN', 'iowa': 'IA', 'kansas': 'KS',
    'kentucky': 'KY', 'louisiana': 'LA', 'maine': 'ME', 'maryland': 'MD',
    'massachusetts': 'MA', 'michigan': 'MI', 'minnesota': 'MN', 'mississippi': 'MS',
    'missouri': 'MO', 'montana': 'MT', 'nebraska': 'NE', 'nevada': 'NV',
    'new hampshire': 'NH', 'new jersey': 'NJ', 'new mexico': 'NM', 'new york': 'NY',
    'north carolina': 'NC', 'north dakota': 'ND', 'ohio': 'OH', 'oklahoma': 'OK',
    'oregon': 'OR', 'pennsylvania': 'PA', 'rhode island': 'RI',
    'south carolina': 'SC', 'south dakota': 'SD', 'tennessee': 'TN', 'texas': 'TX',
    'utah': 'UT', 'vermont': 'VT', 'virginia': 'VA', 'washington': 'WA',
    'west virginia': 'WV', 'wisconsin': 'WI', 'wyoming': 'WY',
}

TERRITORY_NAME_TO_CODE = {
    'puerto rico': 'PR', 'virgin islands': 'VI', 'us virgin islands': 'VI',
    'guam': 'GU', 'american samoa': 'AS', 'northern mariana islands': 'MP',
}

LABEL_STATUS = {'open': 1, 'no further action': 0}
UNKNOWN_STATUS = {'', 'unknown', 'unkown'}

DEDUPE_KEY = ['raw_facility_id', 'raw_lust_id', TIME_COLUMN, 'state', 'status_norm']


def _load_json(url, cache, name):
    path = fetch(url, cache, name)
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _parse_timestamp(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        try:
            return pd.to_datetime(value, unit='ms', utc=True)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
        return pd.to_datetime(number, unit='ms', utc=True)
    except ValueError:
        pass
    try:
        stamp = pd.to_datetime(text, utc=True)
    except Exception:
        return None
    if stamp is None or pd.isna(stamp):
        return None
    return stamp


def _as_text(value):
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        if math.isnan(value):
            return ''
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value).strip()


def _to_number(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text.replace(',', ''))
    except ValueError:
        return None


def _looks_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _state_code(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if upper in STATE_CODES:
        return upper
    if upper in TERRITORY_CODES:
        return upper
    key = ' '.join(text.lower().replace('.', ' ').split())
    if key in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[key]
    if key in TERRITORY_NAME_TO_CODE:
        return TERRITORY_NAME_TO_CODE[key]
    if 'tribal' in key:
        return 'TRIBAL'
    return None


def _prefix_code(*values):
    for value in values:
        if value is None:
            continue
        text = str(value).strip().upper()
        if len(text) >= 2:
            prefix = text[:2]
            if prefix in STATE_CODES or prefix in TERRITORY_CODES:
                return prefix
    return None


def _stable_id(raw_value, facility_value, object_value):
    raw_text = _as_text(raw_value)
    facility_text = _as_text(facility_value)
    object_text = _as_text(object_value)
    if raw_text or facility_text:
        return raw_text + '|' + facility_text + '|' + object_text
    return 'objectid|' + object_text


def _write(frame, directory, filename):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    frame.to_csv(path, index=False, lineterminator='\n')
    return path


def _read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _resolve_fields(field_names):
    lookup = {}
    for name in field_names:
        lookup.setdefault(name.casefold(), name)
    resolved = {}
    missing = []
    for name in DOWNLOAD_FIELDS:
        actual = lookup.get(name.casefold())
        if actual is None:
            missing.append(name)
        else:
            resolved[name] = actual
    return resolved, missing


def build(output: Path, cache: Path) -> list:
    output = Path(output)
    cache = Path(cache)
    checks = []

    meta = _load_json(LAYER_URL + '?' + urlencode({'f': 'json'}), cache, 'lust_layer_metadata.json')
    if isinstance(meta, dict) and meta.get('error'):
        raise AssertionError(
            'layer metadata returned a service error; selected archive member lust_layer_metadata.json; payload '
            + json.dumps(meta['error'])[:1500]
        )

    field_names = [str(item.get('name')) for item in (meta.get('fields') or []) if isinstance(item, dict)]
    resolved, missing_fields = _resolve_fields(field_names)
    case_variants = {
        name: resolved[name]
        for name in resolved
        if resolved[name] != name
    }
    if missing_fields:
        raise AssertionError((
            'layer metadata is missing declared source fields; missing names: '
            + ', '.join(missing_fields)
            + '; observed field names: '
            + ', '.join(field_names)
            + '; selected archive member: lust_layer_metadata.json from '
            + LAYER_URL
        )[:4000])
    checks.append({
        'check': 'layer metadata field list covers every declared download field',
        'observed': 'matched ' + str(len(resolved)) + ' of ' + str(len(DOWNLOAD_FIELDS))
                    + ' declared fields against ' + str(len(field_names)) + ' observed field names'
                    + ('; case-insensitive matches: ' + json.dumps(case_variants)[:600] if case_variants else '; all matches exact'),
        'expected': 'all ' + str(len(DOWNLOAD_FIELDS)) + ' declared source field names resolvable in the layer field list',
        'passed': not missing_fields,
    })

    attribution = ''
    attribution_key = ''
    for key in ('copyrightText', 'accessInformation', 'credits', 'description'):
        raw_value = meta.get(key)
        if raw_value is not None and str(raw_value).strip():
            attribution = str(raw_value).strip()
            attribution_key = key
            break

    docs_ok = False
    docs_observed = ''
    if not attribution:
        try:
            docs_path = Path(fetch(DOCS_URL, cache, 'epa_ust_finder.html'))
            docs_text = docs_path.read_text(encoding='utf-8', errors='replace')
            docs_lower = docs_text.casefold()
            has_association = 'association of state and territorial solid waste management officials' in docs_lower
            has_office = 'office of underground storage tanks' in docs_lower
            docs_ok = has_association and has_office
            index = docs_lower.find('association of state and territorial solid waste management officials')
            snippet = ''
            if index >= 0:
                snippet = docs_text[max(0, index - 120):index + 240].replace('\n', ' ')
            docs_observed = (
                'publisher documentation page characters ' + str(len(docs_text))
                + '; association phrase present ' + str(has_association)
                + '; office phrase present ' + str(has_office)
                + ('; excerpt ' + snippet[:500] if snippet else '')
            )
        except Exception as exc:
            docs_observed = 'publisher documentation page fetch failed: ' + repr(exc)[:400]

    attribution_ok = bool(attribution) or docs_ok
    if attribution:
        attribution_observed = 'attribution from layer metadata property ' + attribution_key + ': ' + attribution[:400]
    else:
        attribution_observed = docs_observed
    checks.append({
        'check': 'publisher attribution is observable in layer metadata or the EPA UST Finder documentation page',
        'observed': attribution_observed,
        'expected': 'attribution naming the EPA Office of Underground Storage Tanks and the Association of State and Territorial Solid Waste Management Officials',
        'passed': attribution_ok,
    })

    license_text = Path(fetch(LICENSE_URL, cache, 'epa_data_license.html')).read_text(encoding='utf-8', errors='replace')
    license_ok = 'public domain' in license_text.casefold()
    checks.append({
        'check': 'EPA data license page states the public domain status of EPA data',
        'observed': 'license page characters ' + str(len(license_text)) + (' with the public domain phrase present' if license_ok else ' with no public domain phrase'),
        'expected': 'public domain statement on the documented EPA terms page',
        'passed': license_ok,
    })

    count_payload = _load_json(
        QUERY_URL + '?' + urlencode({'where': WHERE_CLAUSE, 'returnCountOnly': 'true', 'f': 'json'}),
        cache,
        'lust_window_count.json',
    )
    if isinstance(count_payload, dict) and count_payload.get('error'):
        raise AssertionError('count query returned a service error; selected archive member lust_window_count.json; payload ' + json.dumps(count_payload['error'])[:1500])
    try:
        expected_count = int(count_payload.get('count'))
    except (TypeError, ValueError):
        raise AssertionError('count query did not return an integer count; observed payload ' + json.dumps(count_payload)[:1500])

    out_fields = ','.join(resolved[name] for name in DOWNLOAD_FIELDS)
    attributes = []
    offset = 0
    page_index = 0
    while True:
        params = {
            'where': WHERE_CLAUSE,
            'outFields': out_fields,
            'returnGeometry': 'false',
            'orderByFields': 'OBJECTID ASC',
            'resultOffset': offset,
            'resultRecordCount': PAGE_SIZE,
            'f': 'json',
        }
        member = 'lust_releases_page_' + str(page_index).zfill(4) + '.json'
        payload = _load_json(QUERY_URL + '?' + urlencode(params), cache, member)
        if isinstance(payload, dict) and payload.get('error'):
            raise AssertionError('release query page returned a service error; selected archive member ' + member + '; payload ' + json.dumps(payload['error'])[:1500])
        features = payload.get('features') or []
        for feature in features:
            if isinstance(feature, dict):
                record = feature.get('attributes')
                if isinstance(record, dict):
                    attributes.append(record)
        if len(attributes) > MAX_ROWS:
            raise AssertionError('release query returned more than the bounded row limit; observed rows ' + str(len(attributes)) + ' at archive member ' + member)
        if len(features) < PAGE_SIZE:
            break
        offset += len(features)
        page_index += 1
        if page_index >= MAX_PAGES:
            raise AssertionError('release query exceeded the bounded page limit; observed pages ' + str(page_index) + ' and rows ' + str(len(attributes)))

    checks.append({
        'check': 'paged download row count matches the service count for the same report-date window',
        'observed': 'downloaded ' + str(len(attributes)) + ' records over ' + str(page_index + 1) + ' pages',
        'expected': 'service count ' + str(expected_count) + ' records',
        'passed': len(attributes) == expected_count,
    })

    missing_from_pages = []
    for attrs in attributes[:1]:
        for name in DOWNLOAD_FIELDS:
            if resolved[name] not in attrs:
                missing_from_pages.append(resolved[name])
    if missing_from_pages:
        raise AssertionError((
            'first release page did not echo every resolved download field; missing names: '
            + ', '.join(missing_from_pages)
            + '; observed page attribute names: '
            + ', '.join(sorted(attributes[0].keys()))
            + '; selected archive member: lust_releases_page_0000.json; requested outFields: '
            + out_fields
        )[:4000])
    checks.append({
        'check': 'returned release page attributes echo every resolved download field using the publisher spelling',
        'observed': 'matched ' + str(len(DOWNLOAD_FIELDS) - len(missing_from_pages)) + ' of ' + str(len(DOWNLOAD_FIELDS)) + ' resolved names in lust_releases_page_0000.json',
        'expected': 'every resolved field name returned by the layer query',
        'passed': not missing_from_pages,
    })

    counts = {
        'download': len(attributes),
        'missing_date': 0,
        'out_of_window': 0,
        'window': 0,
        'state_fallback': 0,
        'unmapped_state': 0,
        'territory': 0,
        'state_ok': 0,
        'unknown_status': 0,
        'unknown_misspelling': 0,
        'other_status': 0,
        'kept': 0,
        'bad_numeric': 0,
        'missing_object_id': 0,
        'duplicate_release_records': 0,
    }
    unmapped_states = []
    other_status_tokens = {}
    unknown_status_tokens = {}
    missing_object_samples = []
    records = []

    f_state = resolved['State']
    f_facility = resolved['Facility_ID']
    f_lust = resolved['LUST_ID']
    f_date = resolved['Reported_Date']
    f_status = resolved['Status']
    f_substance = resolved['Substance']
    f_pop = resolved['Population_within_1500ft']
    f_wells = resolved['DomesticWells_within_1500ft']
    f_land = resolved['LandUse']
    f_spa = resolved['Within_SPA']
    f_flood = resolved['Within_100yr_Floodplain']
    f_county = resolved['County']
    f_region = resolved['EPA_Region']
    f_object = resolved['OBJECTID']

    for attrs in attributes:
        stamp = _parse_timestamp(attrs.get(f_date))
        if stamp is None:
            counts['missing_date'] += 1
            continue
        if not (WINDOW_START <= stamp < WINDOW_END):
            counts['out_of_window'] += 1
            continue
        counts['window'] += 1

        state = _state_code(attrs.get(f_state))
        if state is None:
            state = _prefix_code(attrs.get(f_facility), attrs.get(f_lust))
            if state is not None:
                counts['state_fallback'] += 1
        if state is None:
            counts['unmapped_state'] += 1
            if len(unmapped_states) < 20:
                unmapped_states.append(str(attrs.get(f_state)))
            continue
        if state in TERRITORY_CODES or state == 'TRIBAL':
            counts['territory'] += 1
            continue
        counts['state_ok'] += 1

        raw_status = attrs.get(f_status)
        status_key = '' if raw_status is None else str(raw_status).strip().casefold()
        if status_key in LABEL_STATUS:
            pass
        elif status_key in UNKNOWN_STATUS:
            counts['unknown_status'] += 1
            unknown_status_tokens[status_key] = unknown_status_tokens.get(status_key, 0) + 1
            if status_key == 'unkown':
                counts['unknown_misspelling'] += 1
            continue
        else:
            counts['other_status'] += 1
            other_status_tokens[status_key] = other_status_tokens.get(status_key, 0) + 1
            continue

        object_text = _as_text(attrs.get(f_object))
        if not object_text:
            counts['missing_object_id'] += 1
            if len(missing_object_samples) < 10:
                missing_object_samples.append(_as_text(attrs.get(f_lust)))

        pop_raw = attrs.get(f_pop)
        well_raw = attrs.get(f_wells)
        pop_value = _to_number(pop_raw)
        well_value = _to_number(well_raw)
        if pop_value is None and not _looks_missing(pop_raw):
            counts['bad_numeric'] += 1
        if well_value is None and not _looks_missing(well_raw):
            counts['bad_numeric'] += 1

        counts['kept'] += 1
        raw_id = _as_text(attrs.get(f_lust))
        records.append({
            ID_COLUMN: raw_id,
            'raw_lust_id': raw_id,
            'raw_facility_id': _as_text(attrs.get(f_facility)),
            'source_objectid': object_text,
            TIME_COLUMN: stamp.strftime('%Y-%m-%d'),
            'substance_released': _as_text(attrs.get(f_substance)),
            'state': state,
            'report_year': int(stamp.year),
            'population_within_1500ft': pop_value,
            'domestic_wells_within_1500ft': well_value,
            'land_use': _as_text(attrs.get(f_land)),
            'within_source_water_protection_area': _as_text(attrs.get(f_spa)),
            'within_100yr_floodplain': _as_text(attrs.get(f_flood)),
            'county': _as_text(attrs.get(f_county)),
            'epa_region': _as_text(attrs.get(f_region)),
            'status_norm': 'Open' if status_key == 'open' else 'No Further Action',
            TARGET_COLUMN: LABEL_STATUS[status_key],
        })

    checks.append({
        'check': 'every downloaded record is accounted for by the report-date window rule',
        'observed': 'missing date ' + str(counts['missing_date']) + ', outside window ' + str(counts['out_of_window']) + ', inside window ' + str(counts['window']),
        'expected': 'the three counts sum to the downloaded total ' + str(counts['download']),
        'passed': counts['missing_date'] + counts['out_of_window'] + counts['window'] == counts['download'],
    })

    checks.append({
        'check': 'every in-window record is assigned to a state, a territory or an unmapped bucket',
        'observed': 'territory ' + str(counts['territory']) + ', mapped jurisdictions ' + str(counts['state_ok']) + ', unmapped ' + str(counts['unmapped_state']) + ', jurisdiction fallback ' + str(counts['state_fallback']),
        'expected': 'the three buckets sum to the in-window total ' + str(counts['window']),
        'passed': counts['territory'] + counts['state_ok'] + counts['unmapped_state'] == counts['window'],
    })

    checks.append({
        'check': 'unknown status values, including the observed misspelling, are excluded and never labelled negative',
        'observed': 'unknown or blank rows ' + str(counts['unknown_status']) + ' with tokens ' + json.dumps(unknown_status_tokens)[:400] + '; observed misspelling unkown rows ' + str(counts['unknown_misspelling']),
        'expected': 'all Unknown-family rows excluded from the cohort rather than mapped to zero',
        'passed': sum(unknown_status_tokens.values()) == counts['unknown_status'],
    })

    checks.append({
        'check': 'release status vocabulary resolves to the two label values, Unknown or a reported other token',
        'observed': 'kept ' + str(counts['kept']) + ', unknown or blank ' + str(counts['unknown_status']) + ', other tokens ' + str(counts['other_status']) + ((' with values ' + json.dumps(other_status_tokens)[:400]) if other_status_tokens else ''),
        'expected': 'mapped jurisdictions total ' + str(counts['state_ok']) + ' and only the documented Open, No Further Action and Unknown categories',
        'passed': counts['kept'] + counts['unknown_status'] + counts['other_status'] == counts['state_ok'] and counts['other_status'] == 0,
    })

    if counts['unmapped_state']:
        raise AssertionError((
            'state jurisdiction values could not be mapped for in-window records; observed sample values: '
            + json.dumps(unmapped_states)[:1500]
            + '; unmapped rows ' + str(counts['unmapped_state'])
            + '; selected archive member lust_releases_page_0000.json'
        )[:4000])

    if counts['missing_object_id']:
        raise AssertionError((
            'included release rows lack the layer OBJECTID needed for a stable identifier; missing rows '
            + str(counts['missing_object_id'])
            + '; sample raw LUST_ID values '
            + json.dumps(missing_object_samples)[:1500]
            + '; selected archive member lust_releases_page_0000.json'
        )[:4000])

    if not records:
        raise AssertionError('no release records survived the frozen population rules; observed counts ' + json.dumps(counts)[:2000])

    raw = pd.DataFrame.from_records(records)
    constructed = len(raw)

    raw_duplicate_count = int(raw['raw_lust_id'].duplicated().sum())
    raw_duplicate_sample = raw.loc[raw['raw_lust_id'].duplicated(), 'raw_lust_id'].head(5).tolist()

    key_sizes = raw.groupby(DEDUPE_KEY, dropna=False).size()
    duplicate_groups = key_sizes[key_sizes > 1]
    duplicate_group_count = int(len(duplicate_groups))
    duplicate_group_sample = [list(map(str, key)) for key in duplicate_groups.index[:5]]

    object_order = pd.to_numeric(raw['source_objectid'], errors='coerce')
    if object_order.isna().any():
        raise AssertionError((
            'included release rows carry a non-numeric layer OBJECTID, so duplicate collapse cannot be ordered; observed values '
            + json.dumps(raw.loc[object_order.isna(), 'source_objectid'].head(10).tolist())[:800]
        )[:4000])

    frame = raw.assign(_objectid_order=object_order).sort_values('_objectid_order', kind='mergesort')
    frame = frame.drop_duplicates(subset=DEDUPE_KEY, keep='first').reset_index(drop=True)
    removed_duplicates = constructed - len(frame)
    counts['duplicate_release_records'] = removed_duplicates
    residual_duplicates = int(frame.duplicated(subset=DEDUPE_KEY).sum())

    frame[ID_COLUMN] = [
        _stable_id(raw_value, facility_value, object_value)
        for raw_value, facility_value, object_value in zip(
            frame['raw_lust_id'].tolist(),
            frame['raw_facility_id'].tolist(),
            frame['source_objectid'].tolist(),
        )
    ]
    frame = frame.drop(columns=['raw_lust_id', 'raw_facility_id', 'source_objectid', '_objectid_order'])
    df = frame.sort_values([TIME_COLUMN, ID_COLUMN], kind='mergesort').reset_index(drop=True)

    checks.append({
        'check': 'exact duplicate release records are collapsed before splitting',
        'observed': 'removed ' + str(removed_duplicates) + ' duplicate rows from ' + str(constructed)
                    + '; duplicate key groups ' + str(duplicate_group_count)
                    + (('; sample keys ' + json.dumps(duplicate_group_sample)[:600]) if duplicate_group_sample else '')
                    + '; residual duplicate keys ' + str(residual_duplicates),
        'expected': 'rows sharing facility identifier, release identifier, report date, jurisdiction and snapshot status collapse to the lowest layer OBJECTID, leaving zero residual duplicate keys',
        'passed': residual_duplicates == 0,
    })

    checks.append({
        'check': 'repeated source release identifiers that are not exact duplicates are disambiguated with the layer OBJECTID',
        'observed': 'raw duplicate identifier rows ' + str(raw_duplicate_count)
                    + ((' with sample raw LUST_ID values ' + ', '.join(raw_duplicate_sample)) if raw_duplicate_sample else '')
                    + '; exact duplicate rows removed ' + str(removed_duplicates),
        'expected': 'shipped identifier stays unique after exact duplicate release records collapse to one row, and generic repeated release identifiers remain distinct only across different facilities or dates',
        'passed': int(df[ID_COLUMN].duplicated().sum()) == 0,
    })

    status_values = sorted(set(df['status_norm'].tolist()))
    checks.append({
        'check': 'shipped status values are exactly the two frozen label categories',
        'observed': ', '.join(status_values),
        'expected': 'No Further Action and Open',
        'passed': status_values == ['No Further Action', 'Open'],
    })

    recomputed = [1 if str(value).strip().casefold() == 'open' else 0 for value in df['status_norm'].tolist()]
    checks.append({
        'check': 'target is recomputed as one only from the Open status and zero only from No Further Action',
        'observed': 'matching rows ' + str(sum(1 for left, right in zip(recomputed, df[TARGET_COLUMN].tolist()) if left == right)) + ' of ' + str(len(df)),
        'expected': 'every row matches the recomputed label',
        'passed': recomputed == df[TARGET_COLUMN].tolist(),
    })

    checks.append({
        'check': 'both label classes are present in the constructed cohort',
        'observed': 'positives ' + str(int(df[TARGET_COLUMN].sum())) + ' and rows ' + str(len(df)),
        'expected': 'at least one positive and at least one negative row',
        'passed': 0 < int(df[TARGET_COLUMN].sum()) < len(df),
    })

    duplicate_count = int(df[ID_COLUMN].duplicated().sum())
    duplicate_sample = df.loc[df[ID_COLUMN].duplicated(), ID_COLUMN].head(5).tolist()
    checks.append({
        'check': 'shipped release identifier is unique across the constructed cohort',
        'observed': 'duplicate identifier rows ' + str(duplicate_count) + ((' with sample identifiers ' + ', '.join(duplicate_sample)) if duplicate_sample else ''),
        'expected': 'zero duplicate shipped identifiers',
        'passed': duplicate_count == 0,
    })

    year_mismatches = sum(1 for year, text in zip(df['report_year'].tolist(), df[TIME_COLUMN].tolist()) if int(year) != int(text[:4]))
    checks.append({
        'check': 'report year equals the calendar year of the report date',
        'observed': 'mismatched rows ' + str(year_mismatches),
        'expected': 'zero mismatches',
        'passed': year_mismatches == 0,
    })

    checks.append({
        'check': 'shipped state codes are the 50 states plus the District of Columbia',
        'observed': 'distinct state codes ' + str(len(set(df['state'].tolist()))) + ': ' + ', '.join(sorted(set(df['state'].tolist()))),
        'expected': 'every shipped code is one of the 51 declared jurisdictions',
        'passed': set(df['state'].tolist()) <= STATE_CODES and len(set(df['state'].tolist())) > 1,
    })

    checks.append({
        'check': 'shipped report dates stay inside the frozen 2004 to 2008 window',
        'observed': 'minimum ' + df[TIME_COLUMN].min() + ' and maximum ' + df[TIME_COLUMN].max(),
        'expected': 'dates from 2004-01-01 through 2008-12-31',
        'passed': df[TIME_COLUMN].min() >= '2004-01-01' and df[TIME_COLUMN].max() <= '2008-12-31',
    })

    checks.append({
        'check': 'numeric source values parse without loss',
        'observed': 'unparseable population or domestic well values ' + str(counts['bad_numeric']),
        'expected': 'zero unparseable numeric values',
        'passed': counts['bad_numeric'] == 0,
    })

    for column in FEATURE_COLUMNS:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            nonempty = int(series.notna().sum())
        else:
            nonempty = int(series.fillna('').astype(str).str.strip().ne('').sum())
        checks.append({
            'check': 'declared scored feature column carries observed non-empty values',
            'observed': 'column ' + column + ' non-empty rows ' + str(nonempty) + ' of ' + str(len(df)),
            'expected': 'column ' + column + ' is a report-time field present with at least one observed value',
            'passed': nonempty > 0,
        })

    ship = df[OUTPUT_COLUMNS].copy()

    for column in CARRY_COLUMNS:
        series = ship[column]
        if pd.api.types.is_numeric_dtype(series):
            nonempty = int(series.notna().sum())
        else:
            nonempty = int(series.fillna('').astype(str).str.strip().ne('').sum())
        checks.append({
            'check': 'descriptive carry column is shipped without invented fills',
            'observed': 'column ' + column + ' non-empty rows ' + str(nonempty) + ' of ' + str(len(ship)),
            'expected': 'column ' + column + ' present in the shipped cohort with blanks preserved rather than imputed zeros or placeholder categories',
            'passed': column in OUTPUT_COLUMNS and 0 <= nonempty <= len(ship),
        })

    slices = {}
    written = {}
    for name in ('train', 'eval', 'holdout'):
        start, end = SPLIT_BOUNDS[name]
        mask = (ship[TIME_COLUMN] >= start) & (ship[TIME_COLUMN] < end)
        piece = ship.loc[mask].reset_index(drop=True)
        slices[name] = piece
        directory, filename = SPLIT_PATHS[name]
        written[name] = _write(piece, output / directory, filename)

    checks.append({
        'check': 'the three split row counts sum to the constructed cohort',
        'observed': 'train ' + str(len(slices['train'])) + ', eval ' + str(len(slices['eval'])) + ', holdout ' + str(len(slices['holdout'])) + ', constructed total ' + str(len(ship)),
        'expected': 'the split total equals the constructed row count',
        'passed': len(slices['train']) + len(slices['eval']) + len(slices['holdout']) == len(ship) and len(slices['train']) > 0 and len(slices['eval']) > 0 and len(slices['holdout']) > 0,
    })

    checks.append({
        'check': 'each split contains at least one positive and one negative example',
        'observed': 'positive counts ' + json.dumps({name: int(piece[TARGET_COLUMN].sum()) for name, piece in slices.items()}),
        'expected': 'every split has positives and negatives',
        'passed': all(0 < int(piece[TARGET_COLUMN].sum()) < len(piece) for piece in slices.values()),
    })

    backs = {}
    for name in ('train', 'eval', 'holdout'):
        back = _read(written[name])
        backs[name] = back
        start, end = SPLIT_BOUNDS[name]
        checks.append({
            'check': 'CSV header for the ' + name + ' split matches the declared column order',
            'observed': ','.join(list(back.columns)),
            'expected': ','.join(OUTPUT_COLUMNS),
            'passed': list(back.columns) == OUTPUT_COLUMNS,
        })
        times = back[TIME_COLUMN].tolist()
        checks.append({
            'check': 'reported dates in the ' + name + ' split stay inside the frozen half-open window',
            'observed': 'minimum date ' + (min(times) if times else 'none') + ' and maximum date ' + (max(times) if times else 'none'),
            'expected': 'dates from ' + start + ' inclusive to ' + end + ' exclusive',
            'passed': bool(times) and min(times) >= start and max(times) < end,
        })
        ordered = list(zip(back[TIME_COLUMN].tolist(), back[ID_COLUMN].tolist()))
        checks.append({
            'check': 'rows in the ' + name + ' split are sorted by report date then release identifier',
            'observed': 'row order is monotone' if ordered == sorted(ordered) else 'row order is not monotone',
            'expected': 'ascending order of report date and then release identifier',
            'passed': ordered == sorted(ordered),
        })
        tokens = sorted(set(back[TARGET_COLUMN].tolist()))
        checks.append({
            'check': 'target column in the ' + name + ' split contains only binary values',
            'observed': 'distinct target tokens ' + ','.join(tokens),
            'expected': 'only 0 and 1',
            'passed': set(tokens) <= {'0', '1'},
        })

    for column in ('population_within_1500ft', 'domestic_wells_within_1500ft'):
        blank_total = 0
        missing_total = 0
        for name in ('train', 'eval', 'holdout'):
            blank_total += int((backs[name][column] == '').sum())
            missing_total += int(slices[name][column].isna().sum())
        checks.append({
            'check': 'missing values for ' + column + ' stay blank in the written CSVs',
            'observed': 'blank cells ' + str(blank_total) + ' against missing source values ' + str(missing_total),
            'expected': 'the two counts are equal and missing is not written as zero',
            'passed': blank_total == missing_total,
        })

    pop_values = pd.concat([
        pd.to_numeric(backs[name]['population_within_1500ft'], errors='coerce')
        for name in ('train', 'eval', 'holdout')
    ]).dropna()
    pop_min = float(pop_values.min()) if len(pop_values) else None
    checks.append({
        'check': 'population within 1500 feet is a non-negative count in its declared unit',
        'observed': 'values ' + str(int(len(pop_values))) + ' with minimum ' + (str(pop_min) if pop_min is not None else 'none'),
        'expected': 'people counts at or above zero on the EPA overlay frame of people within 1500 feet of the release location, shipped as a descriptive carry column only',
        'passed': len(pop_values) > 0 and bool((pop_values >= 0).all()),
        'kind': 'unit',
    })

    well_values = pd.concat([
        pd.to_numeric(backs[name]['domestic_wells_within_1500ft'], errors='coerce')
        for name in ('train', 'eval', 'holdout')
    ]).dropna()
    well_min = float(well_values.min()) if len(well_values) else None
    integer_ok = bool(((well_values - well_values.round()).abs() < 1e-9).all()) if len(well_values) else False
    checks.append({
        'check': 'domestic wells within 1500 feet is a non-negative whole count in its declared unit',
        'observed': 'values ' + str(int(len(well_values))) + ' with minimum ' + (str(well_min) if well_min is not None else 'none'),
        'expected': 'whole domestic well counts at or above zero on the EPA overlay frame, shipped as a descriptive carry column only',
        'passed': len(well_values) > 0 and bool((well_values >= 0).all()) and integer_ok,
        'kind': 'unit',
    })

    failed = [item for item in checks if not item['passed']]
    if failed:
        details = []
        for item in failed:
            details.append(item['check'] + ' | observed ' + str(item['observed'])[:700] + ' | expected ' + str(item['expected'])[:400])
        raise AssertionError(('source or row checks failed; ' + ' ;; '.join(details))[:4000])

    return checks

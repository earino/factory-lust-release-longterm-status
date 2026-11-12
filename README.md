---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

# Will a reported U.S. leaking underground storage tank release still be unresolved a decade later? Predicting long-run LUST cleanup status from report-time product and site attributes

Will a leaking underground storage tank release reported between 2004 and 2008 still be classified Open, rather than No Further Action, at the 2018-2019 state data snapshot?

Derived from U.S. Environmental Protection Agency, Office of Underground Storage Tanks and Office of Research and Development, with the Association of State and Territorial Solid Waste Management Officials. This is not its official product.

Prediction time: At the release report date recorded by the state program (Reported_Date), when the release first enters the state program record and before any cleanup status is known. Only fields fixed at that instant are scored; spatial overlays attached at the later 2018-2019 compilation ship as descriptive carries.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.8105961663385577,
    "brier": 0.11204408415356631
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.13837729109310853
  },
  "calendar_identity": {
    "auc": 0.5017442015534531,
    "brier": 0.13844349141990345
  }
}
```

## Limitations

- The release layer has no closure date: Status is a snapshot as of each state's 2018-2019 collection, so the label is 'not yet No Further Action at the snapshot', not a fixed-horizon survival outcome.
- Follow-up length differs slightly by report year (about 10-14 years), so report year and horizon are partly confounded; chronological splits preserve the shift rather than removing it.
- State programs differ in what counts as a reportable release, when reporting began, and how No Further Action is granted; state is both a feature and a source of label heterogeneity.
- Release Substance is blank for some records and states, and status Unknown exists (notably Kentucky), so missingness and Unknown exclusions must be modelled explicitly.
- Population, wells, land use, source-water and floodplain fields are current-vintage EPA overlays, not measurements available in 2004-2008; they are slow-changing proxies only.
- The compilation is state-sourced and EPA-standardized; state agencies are credited in the layer metadata, and state-specific terms were not separately researched.
- This is retrospective current-vintage research; it does not establish that the fields were operationally available to a manager at the historical decision instant.
- The release layer has no closure date; Status is a snapshot as of each state 2018-2019 collection, so the label is not yet No Further Action at the snapshot, not a fixed-horizon survival outcome.
- Follow-up length differs by report year, roughly 10 to 14 years, so report year and outcome horizon are partly confounded; the chronological splits preserve that shift rather than removing it.
- Only report-time fields are scored. The five spatial overlays, population and domestic wells within 1500 feet, land use, source water protection area and 100-year floodplain, are attached at the 2018-2019 compilation and the population overlay is derived from 2010 Census blocks, so they postdate the report instant and are shipped as unscored descriptive carry columns instead of features.
- Substance is blank for most records, so the product-type signal is thin; blank values are kept blank rather than imputed, and a fully blank scored feature column fails rather than shipping empty.
- Exact duplicate release records sharing facility identifier, release identifier, report date, jurisdiction and snapshot status are collapsed to the single row with the lowest layer OBJECTID before splitting. This may also merge genuinely separate same-day releases at one facility that share a generic release identifier such as Release 1, which the source cannot distinguish by attributes alone.
- The population and domestic well counts within 1500 feet are continuous counts with many distinct values, so they are declared as high cardinality and must be treated as numeric rather than categorical levels.
- The compilation is state-sourced and EPA-standardised; state agency terms were not separately researched.
- This is retrospective current-vintage research; it does not establish that the fields were operationally available to a manager at the historical decision instant, and the training label timing is unverified.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.

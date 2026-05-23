# Manuscript Reproduction Code

This folder contains the code and public data used for the manuscript:

`Mass-Conserving Simulation of Filter Coffee Extraction from Incomplete Recipe Descriptors`

The repository version in this folder is intended for manuscript review and reproduction. It is separate from the interactive web-service code in `../webpage`.

## Scope

The code implements a mass-conserving porous-bed process simulator for filter coffee extraction. It includes:

- conical brewer geometry for the calibrated case study
- scheduled water input
- finite-volume water and dissolved-solids accounting
- measured PSD conversion from surface-area fraction to mass fraction
- PSD-conditioned D90 closure functions for retained water, hydraulic correction, and dissolved-solids release
- matched in-house calibration analysis
- public recipe incomplete-input reconstruction analysis
- scenario-sensitivity analyses used for manuscript support

The public recipe data are not used for coefficient fitting. Reported public finish time is used only for comparison after simulation.

## Not Included

The first rebuilt core does not include caffeine, electrochemistry, wall bypass, bed deformation, temperature evolution, or filter-specific clogging. These are later modules.

## Installation

Use Python 3.10 or later.

```bash
pip install -r requirements.txt
```

## Main Data Files

- `data/PSD.csv`: measured surface-area PSDs for fine, medium, and coarse grind states
- `data/pour-over data.csv`: matched in-house calibration experiment
- `data/public_data.csv`: public recipe records used for incomplete-input reconstruction

## Reproduce Main Manuscript Outputs

Run commands from this `publiccode` directory.

```bash
python scripts/run_figure3_calibration_d90_functions.py
python scripts/run_calibrated_partitioning_figure4.py
python scripts/make_fig4_calibrated_partitioning_revised.py
python scripts/run_section44_scenario_sensitivity.py
python scripts/run_d90_closure_revision_outputs.py
python scripts/analyze_public_recipe_relative_error.py
```

The scripts write CSV, markdown summaries, and figures to `outputs/`.

## D90-Conditioned Closure

The calibrated D90-conditioned functions are defined in:

```text
scripts/calibrated_v60_model.py
```


These are descriptive closures from the measured coarse, medium, and fine PSD anchors. They are not universal material laws.

## Tests

```bash
python -m unittest discover -s tests
```

The tests check that measured PSD inputs can be loaded, D90-conditioned parameters are finite, the calibrated medium reference simulation runs, mass balances close, cup water does not exceed input water, and inventories remain nonnegative.

## Interpretation

The matched experiment is a calibration result, not independent validation. The public recipe comparison is an incomplete-input reconstruction analysis, not a pass/fail validation test.

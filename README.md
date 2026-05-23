# Reproduction Code

This folder contains the code and public data used for the manuscript:

`Mass-Conserving Simulation of Filter Coffee Extraction from Incomplete Recipe Descriptors`


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


## Installation

Use Python 3.10 or later.

```bash
pip install -r requirements.txt
```


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


## Interpretation

The matched experiment is a calibration result, not independent validation. The public recipe comparison is an incomplete-input reconstruction analysis, not a pass/fail validation test.

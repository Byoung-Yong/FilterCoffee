# Manuscript Reproduction Code

This folder contains the code and public data used for the manuscript:

`Quantifying internal process variables in filter coffee brewing from recipe-level records`

The repository version in this folder is intended for manuscript review and reproduction. 

It provides an interactive web-service at pourover-simulator.vercel.app .

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



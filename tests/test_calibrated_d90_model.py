from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from calibrated_v60_model import (  # noqa: E402
    d90_closure_coefficients_for_scenario,
    d90_closure_config_for_scenario,
    load_calibrated_psd_scenarios,
)
from run_measured_psd_analysis import mass_percentile_diameter_um  # noqa: E402
from v60_physics.parameters import load_config  # noqa: E402
from v60_physics.solver import run_simulation  # noqa: E402


class CalibratedD90ModelTests(unittest.TestCase):
    def test_measured_psd_d90_closure_reference_runs(self) -> None:
        base = load_config(ROOT / "configs" / "default_v60.json")
        scenarios = {scenario.name: scenario for scenario in load_calibrated_psd_scenarios()}
        self.assertEqual({"coarse", "medium", "fine"}, set(scenarios))

        d90_values = {
            name: mass_percentile_diameter_um(scenario, 90.0)
            for name, scenario in scenarios.items()
        }
        self.assertGreater(d90_values["coarse"], d90_values["medium"])
        self.assertGreater(d90_values["medium"], d90_values["fine"])

        medium = scenarios["medium"]
        coefficients = d90_closure_coefficients_for_scenario(medium)
        for value in coefficients.values():
            self.assertGreater(value, 0.0)

        base = replace(
            base,
            geometry=replace(base.geometry, axial_layers=16, radial_bins=4),
            recipe=replace(base.recipe, dt_s=0.1),
        )
        config = d90_closure_config_for_scenario(base, medium)
        result = run_simulation(config, medium.name)
        summary = result.summary

        self.assertTrue(math.isfinite(float(summary["drawdown_time_s"])))
        self.assertLess(float(summary["max_water_residual_abs_g"]), 1e-6)
        self.assertLess(float(summary["max_solids_residual_abs_g"]), 1e-8)
        self.assertLessEqual(
            float(summary["cup_mass_g"]),
            float(summary["total_recipe_water_g"]) + 1e-9,
        )
        self.assertGreaterEqual(float(summary["cup_mass_g"]), 0.0)
        self.assertGreaterEqual(float(summary["bed_water_g"]), 0.0)
        self.assertGreater(float(summary["tds_percent"]), 0.0)
        self.assertGreater(float(summary["ey_percent"]), 0.0)


if __name__ == "__main__":
    unittest.main()

"""Minimal physics-first V60 finite-volume solver.

This module intentionally keeps the first rebuilt model small:

* conical porous bed geometry
* PSD-derived porosity and permeability
* gravity drainage through an axisymmetric finite-volume grid
* simple radial redistribution by saturation gradient
* dissolved-solids release from particles into local pore water
* Darcy-velocity enhancement of local solute release
* advective delivery of liquid and dissolved solids to the cup

It does not include wall bypass, absorbed reservoirs, bed deformation,
caffeine, or electrochemistry. Those must be added only after this core
water and solids balance remains stable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .geometry import Grid, build_grid
from .parameters import ModelConfig, ParticleClass, PourSegment, Scenario
from .psd import (
    characteristic_diameter_m,
    effective_bulk_density_kg_m3,
    effective_permeability_m2,
    log_radius_std,
)


WATER_DENSITY_KG_M3 = 1000.0
GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class DerivedScenario:
    """Scenario properties derived from PSD and material parameters."""

    bulk_density_kg_m3: float
    porosity: float
    permeability_m2: float
    characteristic_diameter_m: float
    psd_log_radius_std: float


@dataclass
class SimulationResult:
    """Container for summary metrics and sampled time series."""

    scenario_name: str
    derived: DerivedScenario
    summary: dict[str, float | str]
    timeseries: list[dict[str, float | str]]


def run_simulation(config: ModelConfig, scenario_name: str) -> SimulationResult:
    """Run one grind or PSD scenario with the configured recipe."""

    scenario = _scenario_by_name(config, scenario_name)
    derived = derive_scenario(config, scenario)
    bed = build_grid(
        coffee_mass_g=config.recipe.coffee_mass_g,
        bulk_density_kg_m3=derived.bulk_density_kg_m3,
        porosity=derived.porosity,
        geometry=config.geometry,
        water_density_kg_m3=config.material.water_density_kg_m3,
    )

    state = _initial_state(config, scenario, bed, derived)
    dt = config.recipe.dt_s
    sample_every = config.recipe.sample_every_s
    total_time = config.recipe.total_time_s
    sample_times = _sample_times(total_time, sample_every)
    next_sample_idx = 0
    series: list[dict[str, float | str]] = []

    t = 0.0
    while t <= total_time + 1e-12:
        while next_sample_idx < len(sample_times) and t >= sample_times[next_sample_idx] - 1e-12:
            series.append(_sample_state(config, scenario_name, bed, derived, state, sample_times[next_sample_idx]))
            next_sample_idx += 1

        if t >= total_time:
            break

        step = min(dt, total_time - t)
        _advance_one_step(config, scenario, bed, derived, state, t, step)
        t += step

    summary = _make_summary(config, scenario_name, bed, derived, state, series)
    return SimulationResult(scenario_name=scenario_name, derived=derived, summary=summary, timeseries=series)


def _scenario_by_name(config: ModelConfig, scenario_name: str) -> Scenario:
    for scenario in config.scenarios:
        if scenario.name == scenario_name:
            return scenario
    known = ", ".join(s.name for s in config.scenarios)
    raise KeyError(f"Unknown scenario {scenario_name!r}. Known scenarios: {known}")


def derive_scenario(config: ModelConfig, scenario: Scenario) -> DerivedScenario:
    """Derive bulk density, porosity, permeability, and PSD spread."""

    sigma = log_radius_std(scenario.particle_classes)
    bulk_density = effective_bulk_density_kg_m3(scenario.particle_classes, config.material)
    raw_porosity = 1.0 - bulk_density / config.material.particle_density_kg_m3
    porosity = min(config.material.maximum_porosity, max(config.material.minimum_porosity, raw_porosity))
    permeability = effective_permeability_m2(
        scenario.particle_classes,
        config.material,
        config.hydraulics.permeability_scale,
    )
    diameter = characteristic_diameter_m(scenario.particle_classes)
    return DerivedScenario(
        bulk_density_kg_m3=bulk_density,
        porosity=porosity,
        permeability_m2=permeability,
        characteristic_diameter_m=diameter,
        psd_log_radius_std=sigma,
    )


def _initial_state(
    config: ModelConfig,
    scenario: Scenario,
    bed: Grid,
    derived: DerivedScenario,
) -> dict[str, np.ndarray | float]:
    nz, nr = bed.cell_volume_m3.shape
    cell_volume = bed.cell_volume_m3
    coffee_mass_cells = cell_volume / cell_volume.sum() * config.recipe.coffee_mass_g
    pore_capacity_g = bed.pore_capacity_g.copy()
    retained_capacity_g = coffee_mass_cells * config.material.retained_water_capacity_g_per_g_coffee
    remaining = np.zeros((len(scenario.particle_classes), nz, nr), dtype=float)
    for class_idx, particle in enumerate(scenario.particle_classes):
        remaining[class_idx] = (
            coffee_mass_cells
            * config.material.extractable_solids_fraction
            * particle.mass_fraction
        )

    return {
        "pool_water_g": 0.0,
        "pore_water_g": np.zeros((nz, nr), dtype=float),
        "retained_water_g": np.zeros((nz, nr), dtype=float),
        "liquid_solids_g": np.zeros((nz, nr), dtype=float),
        "remaining_extractable_g": remaining,
        "cup_water_g": 0.0,
        "cup_solids_g": 0.0,
        "total_input_water_g": 0.0,
        "last_outlet_flow_g_s": 0.0,
        "pore_capacity_g": pore_capacity_g,
        "retained_capacity_g": retained_capacity_g,
        "initial_extractable_g": float(remaining.sum()),
    }


def _advance_one_step(
    config: ModelConfig,
    scenario: Scenario,
    bed: Grid,
    derived: DerivedScenario,
    state: dict[str, np.ndarray | float],
    time_s: float,
    dt_s: float,
) -> None:
    state["last_outlet_flow_g_s"] = 0.0
    _add_pour_to_top(config, bed, state, time_s, dt_s)
    _infiltrate_pool_to_top(state, bed)
    _imbibe_retained_water(config, state, dt_s)
    _release_solids(config, scenario, bed, derived, state, dt_s)
    _vertical_drainage(config, bed, derived, state, dt_s)
    _radial_redistribution(config, state, dt_s)
    _infiltrate_pool_to_top(state, bed)
    _imbibe_retained_water(config, state, dt_s)


def _add_pour_to_top(
    config: ModelConfig,
    bed: Grid,
    state: dict[str, np.ndarray | float],
    time_s: float,
    dt_s: float,
) -> None:
    water_g = _poured_water_g(config.recipe.pours, time_s, dt_s)
    if water_g <= 0.0:
        return
    state["total_input_water_g"] = float(state["total_input_water_g"]) + water_g

    top_idx = bed.cell_volume_m3.shape[0] - 1
    areas = bed.bottom_face_area_m2[top_idx]
    weights = areas / areas.sum()
    pore_water = state["pore_water_g"]
    capacity = state["pore_capacity_g"]
    assert isinstance(pore_water, np.ndarray)
    assert isinstance(capacity, np.ndarray)

    remaining = water_g
    for r_idx, share in enumerate(weights):
        add = water_g * float(share)
        room = max(float(capacity[top_idx, r_idx] - pore_water[top_idx, r_idx]), 0.0)
        accepted = min(add, room)
        pore_water[top_idx, r_idx] += accepted
        remaining -= accepted
    if remaining > 0.0:
        state["pool_water_g"] = float(state["pool_water_g"]) + remaining


def _infiltrate_pool_to_top(state: dict[str, np.ndarray | float], bed: Grid) -> None:
    pool = float(state["pool_water_g"])
    if pool <= 0.0:
        return
    pore_water = state["pore_water_g"]
    capacity = state["pore_capacity_g"]
    assert isinstance(pore_water, np.ndarray)
    assert isinstance(capacity, np.ndarray)
    top_idx = bed.cell_volume_m3.shape[0] - 1
    areas = bed.bottom_face_area_m2[top_idx]
    weights = areas / areas.sum()

    remaining = pool
    for r_idx, share in enumerate(weights):
        target = pool * float(share)
        room = max(float(capacity[top_idx, r_idx] - pore_water[top_idx, r_idx]), 0.0)
        accepted = min(target, room, remaining)
        pore_water[top_idx, r_idx] += accepted
        remaining -= accepted
    state["pool_water_g"] = max(remaining, 0.0)


def _imbibe_retained_water(config: ModelConfig, state: dict[str, np.ndarray | float], dt_s: float) -> None:
    """Move mobile pore water into the immobile retained-water capacity.

    This is a deliberately simple capillary/absorption closure. It separates
    water that can drain by Darcy flow from water that remains associated with
    the wet coffee bed.
    """

    pore_water = state["pore_water_g"]
    retained_water = state["retained_water_g"]
    retained_capacity = state["retained_capacity_g"]
    assert isinstance(pore_water, np.ndarray)
    assert isinstance(retained_water, np.ndarray)
    assert isinstance(retained_capacity, np.ndarray)
    room = np.maximum(retained_capacity - retained_water, 0.0)
    fractional_fill = 1.0 - math.exp(-config.hydraulics.retention_rate_s_inv * dt_s)
    transfer = np.minimum(pore_water, room * fractional_fill)
    pore_water -= transfer
    retained_water += transfer


def _release_solids(
    config: ModelConfig,
    scenario: Scenario,
    bed: Grid,
    derived: DerivedScenario,
    state: dict[str, np.ndarray | float],
    dt_s: float,
) -> None:
    pore_water = state["pore_water_g"]
    retained_water = state["retained_water_g"]
    liquid_solids = state["liquid_solids_g"]
    capacity = state["pore_capacity_g"]
    retained_capacity = state["retained_capacity_g"]
    remaining = state["remaining_extractable_g"]
    assert isinstance(pore_water, np.ndarray)
    assert isinstance(retained_water, np.ndarray)
    assert isinstance(liquid_solids, np.ndarray)
    assert isinstance(capacity, np.ndarray)
    assert isinstance(retained_capacity, np.ndarray)
    assert isinstance(remaining, np.ndarray)

    local_liquid = pore_water + retained_water
    total_liquid_capacity = capacity + retained_capacity
    saturation = np.divide(local_liquid, total_liquid_capacity, out=np.zeros_like(local_liquid), where=total_liquid_capacity > 0.0)
    local_concentration = np.divide(liquid_solids, local_liquid, out=np.zeros_like(liquid_solids), where=local_liquid > 1e-12)
    concentration_factor = np.clip(
        1.0 - local_concentration / config.release.saturation_concentration_g_per_g_water,
        0.0,
        1.0,
    )
    wet_factor = np.clip(saturation, 0.0, 1.0)
    velocity = _local_interstitial_velocity_m_s(config, bed, derived, pore_water, capacity)
    flow_factor = _release_flow_enhancement(config, velocity)

    for class_idx, particle in enumerate(scenario.particle_classes):
        rate = _release_rate_s_inv(config, particle)
        fraction = 1.0 - math.exp(-rate * dt_s)
        effective_fraction = np.minimum(1.0, fraction * wet_factor * concentration_factor * flow_factor)
        released = remaining[class_idx] * effective_fraction
        remaining[class_idx] -= released
        liquid_solids += released


def _local_interstitial_velocity_m_s(
    config: ModelConfig,
    bed: Grid,
    derived: DerivedScenario,
    pore_water: np.ndarray,
    capacity: np.ndarray,
) -> np.ndarray:
    saturation = np.divide(pore_water, capacity, out=np.zeros_like(pore_water), where=capacity > 0.0)
    mobile_sat = np.maximum(
        (saturation - config.hydraulics.residual_saturation)
        / max(1.0 - config.hydraulics.residual_saturation, 1e-12),
        0.0,
    )
    darcy_velocity = (
        derived.permeability_m2
        * WATER_DENSITY_KG_M3
        * GRAVITY_M_S2
        / config.material.water_viscosity_pa_s
        * mobile_sat ** config.hydraulics.relative_permeability_exponent
    )
    porosity = max(derived.porosity, 1e-12)
    return darcy_velocity / porosity


def _release_flow_enhancement(config: ModelConfig, velocity_m_s: np.ndarray) -> np.ndarray:
    coupling = config.release.velocity_extraction_coupling
    if coupling <= 0.0:
        return np.ones_like(velocity_m_s)
    reference = max(config.release.reference_interstitial_velocity_m_s, 1e-12)
    normalized = np.maximum(velocity_m_s, 0.0) / reference
    return 1.0 + coupling * np.minimum(3.0, np.sqrt(normalized))


def _vertical_drainage(
    config: ModelConfig,
    bed: Grid,
    derived: DerivedScenario,
    state: dict[str, np.ndarray | float],
    dt_s: float,
) -> None:
    pore_water = state["pore_water_g"]
    retained_water = state["retained_water_g"]
    liquid_solids = state["liquid_solids_g"]
    capacity = state["pore_capacity_g"]
    assert isinstance(pore_water, np.ndarray)
    assert isinstance(retained_water, np.ndarray)
    assert isinstance(liquid_solids, np.ndarray)
    assert isinstance(capacity, np.ndarray)

    nz, nr = pore_water.shape
    for z_idx in range(nz):
        for r_idx in range(nr):
            sat = pore_water[z_idx, r_idx] / capacity[z_idx, r_idx] if capacity[z_idx, r_idx] > 0 else 0.0
            mobile_sat = max(
                (sat - config.hydraulics.residual_saturation)
                / max(1.0 - config.hydraulics.residual_saturation, 1e-12),
                0.0,
            )
            if mobile_sat <= 0.0:
                continue
            velocity_m_s = (
                derived.permeability_m2
                * WATER_DENSITY_KG_M3
                * GRAVITY_M_S2
                / config.material.water_viscosity_pa_s
                * mobile_sat ** config.hydraulics.relative_permeability_exponent
            )
            proposed = (
                velocity_m_s
                * bed.bottom_face_area_m2[z_idx, r_idx]
                * WATER_DENSITY_KG_M3
                * 1000.0
                * dt_s
            )
            residual_water = capacity[z_idx, r_idx] * config.hydraulics.residual_saturation
            movable = max(float(pore_water[z_idx, r_idx] - residual_water), 0.0)
            transfer = min(proposed, movable)
            if transfer <= 0.0:
                continue

            local_liquid = pore_water[z_idx, r_idx] + retained_water[z_idx, r_idx]
            solids_transfer = _solute_with_water(liquid_solids[z_idx, r_idx], local_liquid, transfer)
            pore_water[z_idx, r_idx] -= transfer
            liquid_solids[z_idx, r_idx] -= solids_transfer
            if z_idx == 0:
                state["cup_water_g"] = float(state["cup_water_g"]) + transfer
                state["cup_solids_g"] = float(state["cup_solids_g"]) + solids_transfer
                state["last_outlet_flow_g_s"] = float(state["last_outlet_flow_g_s"]) + transfer / dt_s
            else:
                receiver_room = max(float(capacity[z_idx - 1, r_idx] - pore_water[z_idx - 1, r_idx]), 0.0)
                accepted = min(transfer, receiver_room)
                rejected = transfer - accepted
                solids_accepted = solids_transfer * (accepted / transfer) if transfer > 0 else 0.0
                solids_rejected = solids_transfer - solids_accepted
                pore_water[z_idx - 1, r_idx] += accepted
                liquid_solids[z_idx - 1, r_idx] += solids_accepted
                if rejected > 0.0:
                    pore_water[z_idx, r_idx] += rejected
                    liquid_solids[z_idx, r_idx] += solids_rejected


def _radial_redistribution(
    config: ModelConfig,
    state: dict[str, np.ndarray | float],
    dt_s: float,
) -> None:
    pore_water = state["pore_water_g"]
    retained_water = state["retained_water_g"]
    liquid_solids = state["liquid_solids_g"]
    capacity = state["pore_capacity_g"]
    assert isinstance(pore_water, np.ndarray)
    assert isinstance(retained_water, np.ndarray)
    assert isinstance(liquid_solids, np.ndarray)
    assert isinstance(capacity, np.ndarray)

    nz, nr = pore_water.shape
    for z_idx in range(nz):
        for left in range(nr - 1):
            right = left + 1
            sat_l = pore_water[z_idx, left] / capacity[z_idx, left] if capacity[z_idx, left] > 0 else 0.0
            sat_r = pore_water[z_idx, right] / capacity[z_idx, right] if capacity[z_idx, right] > 0 else 0.0
            delta = sat_l - sat_r
            if abs(delta) < 1e-12:
                continue
            donor = left if delta > 0.0 else right
            receiver = right if delta > 0.0 else left
            exchange_capacity = 0.5 * (capacity[z_idx, donor] + capacity[z_idx, receiver])
            proposed = config.hydraulics.radial_exchange_rate_s_inv * abs(delta) * exchange_capacity * dt_s
            residual_water = capacity[z_idx, donor] * config.hydraulics.residual_saturation
            movable = max(float(pore_water[z_idx, donor] - residual_water), 0.0)
            room = max(float(capacity[z_idx, receiver] - pore_water[z_idx, receiver]), 0.0)
            transfer = min(float(proposed), movable, room)
            if transfer <= 0.0:
                continue
            local_liquid = pore_water[z_idx, donor] + retained_water[z_idx, donor]
            solids_transfer = _solute_with_water(liquid_solids[z_idx, donor], local_liquid, transfer)
            pore_water[z_idx, donor] -= transfer
            liquid_solids[z_idx, donor] -= solids_transfer
            pore_water[z_idx, receiver] += transfer
            liquid_solids[z_idx, receiver] += solids_transfer


def _release_rate_s_inv(config: ModelConfig, particle: ParticleClass) -> float:
    radius = particle.radius_um * 1e-6
    ref = config.release.reference_radius_um * 1e-6
    return (
        config.release.diffusion_rate_ref_s_inv * (ref / radius) ** 2
        + config.release.surface_rate_ref_s_inv * (ref / radius)
    )


def _solute_with_water(solids_g: float, water_g: float, transfer_water_g: float) -> float:
    if water_g <= 1e-12 or transfer_water_g <= 0.0:
        return 0.0
    return min(solids_g, solids_g * transfer_water_g / water_g)


def _inlet_flow_g_s(pours: Iterable[PourSegment], time_s: float) -> float:
    for segment in pours:
        if segment.start_s <= time_s < segment.end_s:
            return segment.flow_g_s
    return 0.0


def _poured_water_g(pours: Iterable[PourSegment], time_s: float, dt_s: float) -> float:
    step_start = time_s
    step_end = time_s + dt_s
    water = 0.0
    for segment in pours:
        overlap_start = max(step_start, segment.start_s)
        overlap_end = min(step_end, segment.end_s)
        overlap = max(overlap_end - overlap_start, 0.0)
        water += segment.flow_g_s * overlap
    return water


def _sample_times(total_time_s: float, sample_every_s: float) -> list[float]:
    count = int(math.floor(total_time_s / sample_every_s))
    values = [round(i * sample_every_s, 10) for i in range(count + 1)]
    if values[-1] < total_time_s:
        values.append(total_time_s)
    return values


def _sample_state(
    config: ModelConfig,
    scenario_name: str,
    bed: Grid,
    derived: DerivedScenario,
    state: dict[str, np.ndarray | float],
    time_s: float,
) -> dict[str, float | str]:
    pore_water = state["pore_water_g"]
    retained_water = state["retained_water_g"]
    liquid_solids = state["liquid_solids_g"]
    remaining = state["remaining_extractable_g"]
    assert isinstance(pore_water, np.ndarray)
    assert isinstance(retained_water, np.ndarray)
    assert isinstance(liquid_solids, np.ndarray)
    assert isinstance(remaining, np.ndarray)
    initial_extractable = float(state["initial_extractable_g"])
    cup_water = float(state["cup_water_g"])
    cup_solids = float(state["cup_solids_g"])
    total_input = float(state["total_input_water_g"])
    pool = float(state["pool_water_g"])
    mobile_water = float(pore_water.sum())
    immobile_water = float(retained_water.sum())
    released_solids = initial_extractable - float(remaining.sum())
    water_residual = total_input - pool - mobile_water - immobile_water - cup_water
    solids_residual = initial_extractable - float(remaining.sum()) - float(liquid_solids.sum()) - cup_solids

    return {
        "time_s": time_s,
        "scenario": scenario_name,
        "inlet_flow_g_s": _inlet_flow_g_s(config.recipe.pours, time_s),
        "outlet_flow_g_s": float(state["last_outlet_flow_g_s"]),
        "total_input_water_g": total_input,
        "pool_water_g": pool,
        "pore_water_g": mobile_water,
        "retained_water_g": immobile_water,
        "bed_water_g": mobile_water + immobile_water,
        "cup_water_g": cup_water,
        "cup_solids_g": cup_solids,
        "bed_liquid_solids_g": float(liquid_solids.sum()),
        "remaining_extractable_solids_g": float(remaining.sum()),
        "initial_extractable_solids_g": initial_extractable,
        "tds_percent": 100.0 * cup_solids / cup_water if cup_water > 0.0 else 0.0,
        "ey_percent": 100.0 * cup_solids / config.recipe.coffee_mass_g,
        "released_solids_percent": 100.0 * released_solids / initial_extractable if initial_extractable > 0 else 0.0,
        "water_residual_g": water_residual,
        "solids_residual_g": solids_residual,
        "bed_height_mm": bed.bed_height_m * 1000.0,
        "porosity": derived.porosity,
        "permeability_m2": derived.permeability_m2,
    }


def _make_summary(
    config: ModelConfig,
    scenario_name: str,
    bed: Grid,
    derived: DerivedScenario,
    state: dict[str, np.ndarray | float],
    series: list[dict[str, float | str]],
) -> dict[str, float | str]:
    final = series[-1]
    cup_water = float(final["cup_water_g"])
    cup_solids = float(final["cup_solids_g"])
    initial_extractable = float(state["initial_extractable_g"])
    remaining = state["remaining_extractable_g"]
    assert isinstance(remaining, np.ndarray)
    released = initial_extractable - float(remaining.sum())
    drawdown_time = _drawdown_time(series, config.recipe.pours[-1].end_s)
    return {
        "scenario": scenario_name,
        "coffee_mass_g": config.recipe.coffee_mass_g,
        "total_recipe_water_g": config.recipe.total_water_g,
        "total_input_water_g": (
            float(final["cup_water_g"])
            + float(final["pore_water_g"])
            + float(final["retained_water_g"])
            + float(final["pool_water_g"])
        ),
        "cup_mass_g": cup_water,
        "tds_percent": 100.0 * cup_solids / cup_water if cup_water > 0.0 else 0.0,
        "ey_percent": 100.0 * cup_solids / config.recipe.coffee_mass_g,
        "cup_dissolved_solids_g": cup_solids,
        "released_solids_g": released,
        "released_solids_percent": 100.0 * released / initial_extractable if initial_extractable > 0.0 else 0.0,
        "retained_pore_water_g": float(final["pore_water_g"]),
        "retained_immobile_water_g": float(final["retained_water_g"]),
        "bed_water_g": float(final["bed_water_g"]),
        "pool_water_g": float(final["pool_water_g"]),
        "bed_height_mm": bed.bed_height_m * 1000.0,
        "bulk_density_kg_m3": derived.bulk_density_kg_m3,
        "porosity": derived.porosity,
        "permeability_m2": derived.permeability_m2,
        "characteristic_diameter_um": derived.characteristic_diameter_m * 1e6,
        "psd_log_radius_std": derived.psd_log_radius_std,
        "drawdown_time_s": drawdown_time,
        "drawdown_after_final_pour_s": drawdown_time - config.recipe.pours[-1].end_s if math.isfinite(drawdown_time) else float("nan"),
        "max_pool_water_g": max(float(row["pool_water_g"]) for row in series),
        "max_water_residual_abs_g": max(abs(float(row["water_residual_g"])) for row in series),
        "max_solids_residual_abs_g": max(abs(float(row["solids_residual_g"])) for row in series),
    }


def _drawdown_time(series: list[dict[str, float | str]], final_pour_end_s: float) -> float:
    hold_time_s = 10.0
    flow_threshold_g_s = 0.02
    pool_threshold_g = 0.1
    for index, row in enumerate(series):
        start_time = float(row["time_s"])
        if start_time < final_pour_end_s:
            continue
        if float(row["pool_water_g"]) > pool_threshold_g:
            continue
        end_time = start_time + hold_time_s
        window = [
            later
            for later in series[index:]
            if start_time <= float(later["time_s"]) <= end_time + 1e-12
        ]
        if not window or float(window[-1]["time_s"]) < end_time - 1e-9:
            continue
        if all(float(later["outlet_flow_g_s"]) < flow_threshold_g_s for later in window):
            return start_time
    return float("nan")

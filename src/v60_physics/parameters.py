"""Configuration dataclasses for the rebuilt V60 model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PourSegment:
    start_s: float
    end_s: float
    water_g: float

    @property
    def flow_g_s(self) -> float:
        duration = self.end_s - self.start_s
        if duration <= 0.0:
            raise ValueError("Pour segment duration must be positive.")
        return self.water_g / duration


@dataclass(frozen=True)
class Recipe:
    coffee_mass_g: float
    total_time_s: float
    dt_s: float
    sample_every_s: float
    pours: tuple[PourSegment, ...]

    @property
    def total_water_g(self) -> float:
        return sum(segment.water_g for segment in self.pours)


@dataclass(frozen=True)
class GeometryConfig:
    bottom_radius_m: float
    cone_half_angle_deg: float
    axial_layers: int
    radial_bins: int


@dataclass(frozen=True)
class MaterialConfig:
    particle_density_kg_m3: float
    bulk_density_kg_m3: float
    packing_breadth_factor: float
    minimum_porosity: float
    maximum_porosity: float
    extractable_solids_fraction: float
    retained_water_capacity_g_per_g_coffee: float
    water_density_kg_m3: float
    water_viscosity_pa_s: float


@dataclass(frozen=True)
class HydraulicsConfig:
    permeability_scale: float
    residual_saturation: float
    relative_permeability_exponent: float
    inlet_distribution_exponent: float
    radial_exchange_rate_s_inv: float
    retention_rate_s_inv: float


@dataclass(frozen=True)
class ReleaseConfig:
    reference_radius_um: float
    diffusion_rate_ref_s_inv: float
    surface_rate_ref_s_inv: float
    saturation_concentration_g_per_g_water: float
    velocity_extraction_coupling: float = 0.0
    reference_interstitial_velocity_m_s: float = 0.002


@dataclass(frozen=True)
class ParticleClass:
    radius_um: float
    mass_fraction: float


@dataclass(frozen=True)
class Scenario:
    name: str
    particle_classes: tuple[ParticleClass, ...]


@dataclass(frozen=True)
class ModelConfig:
    recipe: Recipe
    geometry: GeometryConfig
    material: MaterialConfig
    hydraulics: HydraulicsConfig
    release: ReleaseConfig
    scenarios: tuple[Scenario, ...]


def _require_keys(data: dict, keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing keys in {context}: {', '.join(missing)}")


def _load_pours(rows: list[dict]) -> tuple[PourSegment, ...]:
    pours = []
    for row in rows:
        _require_keys(row, ("start_s", "end_s", "water_g"), "recipe.pours")
        pours.append(
            PourSegment(
                start_s=float(row["start_s"]),
                end_s=float(row["end_s"]),
                water_g=float(row["water_g"]),
            )
        )
    return tuple(pours)


def _load_scenario(row: dict) -> Scenario:
    _require_keys(row, ("name", "particle_classes"), "scenario")
    classes = []
    total_fraction = 0.0
    for item in row["particle_classes"]:
        _require_keys(item, ("radius_um", "mass_fraction"), f"scenario {row['name']}")
        particle = ParticleClass(
            radius_um=float(item["radius_um"]),
            mass_fraction=float(item["mass_fraction"]),
        )
        if particle.radius_um <= 0.0 or particle.mass_fraction < 0.0:
            raise ValueError(f"Invalid particle class in scenario {row['name']}.")
        total_fraction += particle.mass_fraction
        classes.append(particle)
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError(f"Mass fractions for scenario {row['name']} must sum to 1.")
    return Scenario(name=str(row["name"]), particle_classes=tuple(classes))


def load_config(path: str | Path) -> ModelConfig:
    """Load a model configuration from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    recipe_data = data["recipe"]
    geometry_data = data["geometry"]
    material_data = data["material"]
    hydraulics_data = data["hydraulics"]
    release_data = data["release"]
    recipe = Recipe(
        coffee_mass_g=float(recipe_data["coffee_mass_g"]),
        total_time_s=float(recipe_data["total_time_s"]),
        dt_s=float(recipe_data["dt_s"]),
        sample_every_s=float(recipe_data["sample_every_s"]),
        pours=_load_pours(recipe_data["pours"]),
    )
    geometry = GeometryConfig(
        bottom_radius_m=float(geometry_data["bottom_radius_m"]),
        cone_half_angle_deg=float(geometry_data["cone_half_angle_deg"]),
        axial_layers=int(geometry_data["axial_layers"]),
        radial_bins=int(geometry_data["radial_bins"]),
    )
    material = MaterialConfig(
        particle_density_kg_m3=float(material_data["particle_density_kg_m3"]),
        bulk_density_kg_m3=float(material_data["bulk_density_kg_m3"]),
        packing_breadth_factor=float(material_data["packing_breadth_factor"]),
        minimum_porosity=float(material_data["minimum_porosity"]),
        maximum_porosity=float(material_data["maximum_porosity"]),
        extractable_solids_fraction=float(material_data["extractable_solids_fraction"]),
        retained_water_capacity_g_per_g_coffee=float(
            material_data["retained_water_capacity_g_per_g_coffee"]
        ),
        water_density_kg_m3=float(material_data["water_density_kg_m3"]),
        water_viscosity_pa_s=float(material_data["water_viscosity_pa_s"]),
    )
    hydraulics = HydraulicsConfig(
        permeability_scale=float(hydraulics_data["permeability_scale"]),
        residual_saturation=float(hydraulics_data["residual_saturation"]),
        relative_permeability_exponent=float(hydraulics_data["relative_permeability_exponent"]),
        inlet_distribution_exponent=float(hydraulics_data["inlet_distribution_exponent"]),
        radial_exchange_rate_s_inv=float(hydraulics_data["radial_exchange_rate_s_inv"]),
        retention_rate_s_inv=float(hydraulics_data["retention_rate_s_inv"]),
    )
    release = ReleaseConfig(
        reference_radius_um=float(release_data["reference_radius_um"]),
        diffusion_rate_ref_s_inv=float(release_data["diffusion_rate_ref_s_inv"]),
        surface_rate_ref_s_inv=float(release_data["surface_rate_ref_s_inv"]),
        saturation_concentration_g_per_g_water=float(
            release_data["saturation_concentration_g_per_g_water"]
        ),
        velocity_extraction_coupling=float(release_data.get("velocity_extraction_coupling", 0.0)),
        reference_interstitial_velocity_m_s=float(
            release_data.get("reference_interstitial_velocity_m_s", 0.002)
        ),
    )
    scenarios = tuple(_load_scenario(row) for row in data["scenarios"])
    return ModelConfig(
        recipe=recipe,
        geometry=geometry,
        material=material,
        hydraulics=hydraulics,
        release=release,
        scenarios=scenarios,
    )

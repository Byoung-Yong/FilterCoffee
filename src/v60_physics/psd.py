"""Particle size distribution utilities."""

from __future__ import annotations

import math

from .parameters import MaterialConfig, ParticleClass


def log_radius_std(classes: tuple[ParticleClass, ...]) -> float:
    mean = sum(p.mass_fraction * math.log(p.radius_um) for p in classes)
    variance = sum(p.mass_fraction * (math.log(p.radius_um) - mean) ** 2 for p in classes)
    return math.sqrt(max(0.0, variance))


def effective_bulk_density_kg_m3(classes: tuple[ParticleClass, ...], material: MaterialConfig) -> float:
    sigma = log_radius_std(classes)
    return material.bulk_density_kg_m3 * (1.0 + material.packing_breadth_factor * sigma)


def porosity(classes: tuple[ParticleClass, ...], material: MaterialConfig) -> float:
    raw = 1.0 - effective_bulk_density_kg_m3(classes, material) / material.particle_density_kg_m3
    return min(material.maximum_porosity, max(material.minimum_porosity, raw))


def characteristic_diameter_m(classes: tuple[ParticleClass, ...]) -> float:
    """Surface-weighted harmonic diameter for a mass-fraction PSD."""

    inverse_diameter = 0.0
    for particle in classes:
        diameter_m = 2.0 * particle.radius_um * 1e-6
        inverse_diameter += particle.mass_fraction / diameter_m
    return 1.0 / inverse_diameter


def effective_permeability_m2(
    classes: tuple[ParticleClass, ...],
    material: MaterialConfig,
    permeability_scale: float,
) -> float:
    eps = porosity(classes, material)
    d_char = characteristic_diameter_m(classes)
    return permeability_scale * (d_char**2 * eps**3) / (180.0 * (1.0 - eps) ** 2)


def release_rate_s_inv(radius_um: float, reference_radius_um: float, k_diff: float, k_surf: float) -> float:
    ratio = reference_radius_um / radius_um
    return k_diff * ratio**2 + k_surf * ratio


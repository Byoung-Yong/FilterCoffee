"""Conical V60-like bed geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .parameters import GeometryConfig


@dataclass(frozen=True)
class Grid:
    bed_height_m: float
    layer_edges_m: np.ndarray
    layer_centers_m: np.ndarray
    outer_radius_m: np.ndarray
    radial_edges_m: np.ndarray
    cell_volume_m3: np.ndarray
    pore_capacity_g: np.ndarray
    bottom_face_area_m2: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.cell_volume_m3.shape


def frustum_volume_m3(height_m: float, bottom_radius_m: float, half_angle_rad: float) -> float:
    top_radius = bottom_radius_m + height_m * math.tan(half_angle_rad)
    return math.pi * height_m * (
        bottom_radius_m**2 + bottom_radius_m * top_radius + top_radius**2
    ) / 3.0


def solve_bed_height_m(volume_m3: float, bottom_radius_m: float, half_angle_rad: float) -> float:
    lo = 0.0
    hi = 0.5
    while frustum_volume_m3(hi, bottom_radius_m, half_angle_rad) < volume_m3:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if frustum_volume_m3(mid, bottom_radius_m, half_angle_rad) < volume_m3:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_grid(
    coffee_mass_g: float,
    bulk_density_kg_m3: float,
    porosity: float,
    geometry: GeometryConfig,
    water_density_kg_m3: float,
) -> Grid:
    """Build an axisymmetric finite-volume grid for a conical frustum bed."""

    bed_volume_m3 = (coffee_mass_g / 1000.0) / bulk_density_kg_m3
    half_angle = math.radians(geometry.cone_half_angle_deg)
    bed_height = solve_bed_height_m(bed_volume_m3, geometry.bottom_radius_m, half_angle)
    nz = geometry.axial_layers
    nr = geometry.radial_bins
    layer_edges = np.linspace(0.0, bed_height, nz + 1)
    layer_centers = 0.5 * (layer_edges[:-1] + layer_edges[1:])
    dz = bed_height / nz
    outer_radius = geometry.bottom_radius_m + layer_centers * math.tan(half_angle)
    radial_edges = np.zeros((nz, nr + 1), dtype=float)
    cell_volume = np.zeros((nz, nr), dtype=float)
    bottom_area = np.zeros((nz, nr), dtype=float)
    tan_theta = math.tan(half_angle)
    for iz in range(nz):
        fraction_edges = np.linspace(0.0, 1.0, nr + 1)
        radial_edges[iz, :] = fraction_edges * outer_radius[iz]
        z0 = layer_edges[iz]
        z1 = layer_edges[iz + 1]
        r0 = geometry.bottom_radius_m + z0 * tan_theta
        r1 = geometry.bottom_radius_m + z1 * tan_theta
        layer_volume = math.pi * dz * (r0**2 + r0 * r1 + r1**2) / 3.0
        for ir in range(nr):
            annulus_fraction = fraction_edges[ir + 1] ** 2 - fraction_edges[ir] ** 2
            cell_volume[iz, ir] = layer_volume * annulus_fraction
            bottom_area[iz, ir] = math.pi * r0**2 * annulus_fraction
    pore_capacity = cell_volume * porosity * water_density_kg_m3 * 1000.0
    return Grid(
        bed_height_m=bed_height,
        layer_edges_m=layer_edges,
        layer_centers_m=layer_centers,
        outer_radius_m=outer_radius,
        radial_edges_m=radial_edges,
        cell_volume_m3=cell_volume,
        pore_capacity_g=pore_capacity,
        bottom_face_area_m2=bottom_area,
    )

# -*- coding: utf-8 -*-
"""
Gaia-like local-plane along-scan astrometry simulator.

This script reads a GOST scanning-geometry CSV and generates the synthetic
along-scan observable used by the AL likelihood model:

    AL = (alpha0 + mu_alpha* dt + orbital_alpha*) sin(psi)
       + (delta0 + mu_delta  dt + orbital_delta ) cos(psi)
       + parallax * parallaxFactorAlongScan
       + noise


    output:
    row0 = [AL_start_row, N_al, floor(t_ref_jd_tcb), fractional_t_ref_jd_tcb, ...]
    data row = [
        obs_time_tcb,
        centroid_pos_al_mas,
        centroid_pos_error_al_mas,
        scanAngle_rad,
        sin_psi,
        cos_psi,
        parallaxFactorAlongScan,
        outlier_flag,
        synthetic_transit_id,
        ccd_row,
        fov_code
    ]
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# User configuration: this is normally the only section that needs editing
# =============================================================================

# Path to the GOST scanning-geometry CSV.
GOST_CSV = r"/path/to/user_data/gost_HD164604.csv"

# Output directory and target label.
OUTPUT_DIR = r"/path/to/user_data/gaia_al_output_HD164604"
TARGET_NAME = "HD164604"

# Random seed for reproducible noise realizations.
RANDOM_SEED = 64

# Reference epoch. The default J2016.0 corresponds to JD(TCB)=2457389.0.
USE_CUSTOM_T_REF_JD_TCB = False
REF_EPOCH_JYEAR_TCB = 2016.0
CUSTOM_T_REF_JD_TCB = 2457389.0

# Planet phase convention:
#   "tau" : set tau_fraction for each planet; it is the orbital phase at t_ref measured from the previous periastron passage.
#   "T0"  : set T0_jd for each planet; the code converts it to tau_fraction at t_ref.
PHASE_PARAMETER = "T0"

# Astrometric signal scale convention:
#   "alpha_ast_mas" : set the angular semimajor axis of the stellar reflex motion directly, in mas.
#   "K_mps"         : set the RV semi-amplitude K in m/s; the code converts it to alpha_ast_mas.
#   "mass_mjup"     : set the companion mass in Mjup; the code converts it to alpha_ast_mas.
SCALE_PARAMETER = "K_mps"

# Stellar mass, required only when SCALE_PARAMETER="mass_mjup".
MSTAR_MSUN = 0.77

# Five-parameter single-star astrometric values at t_ref. pmra_star_masyr is mu_alpha*, so do not multiply it by cos(dec) again.
# For t_ref=J2016.0, these quantities can be queried from the Gaia Archive, for example:
# SELECT
#     'jyear' AS epoch_format,
#     g.ref_epoch AS epoch_value,
#     'tcb' AS epoch_scale,
#     g.ra AS ra_init,
#     g.dec AS dec_init,
#     g.pmra AS pm_ra_star_masyr,
#     g.pmdec AS pm_dec_masyr,
#     g.parallax AS parallax_mas,
#     g.radial_velocity AS rv_sys_kms,
#     g.phot_g_mean_mag AS Gmag,
#     ap.mass_flame AS Mstar_Msun
# FROM gaiadr3.gaia_source AS g
# LEFT JOIN gaiadr3.astrophysical_parameters AS ap
#     ON g.source_id = ap.source_id
# WHERE g.source_id =

STAR = dict(
    # Catalog reference position at t_ref.
    # ra_ref_deg is the ordinary Gaia right ascension alpha.
    # It is not alpha*cos(dec).
    # dec_ref_deg is the ordinary Gaia declination.
    ra_ref_deg=270.7787122020918,
    dec_ref_deg=-28.560840236950646,
    # Tangent-plane position offsets at t_ref.
    dalpha0_star_mas=0.0,   # Initial tangent-plane offset Delta alpha* at t_ref, i.e. Delta alpha cos(delta).
    ddelta0_mas=0.0,   # Initial tangent-plane offset Delta delta at t_ref.
    # If both offsets are zero, ra_ref_deg and dec_ref_deg define the local tangent-plane origin at t_ref.
    # Non-zero offsets mean that the star is displaced from that origin in the local tangent plane at t_ref.
    pmra_star_masyr=-34.65794633176605,
    pmdec_masyr=-42.25317818384521,
    parallax_mas=24.986716148982897,
)

# Planet parameters.
# omega_deg is the companion-relative argument of periastron. The code uses -alpha_ast_mas to generate the stellar reflex motion.
# If PHASE_PARAMETER="tau", provide tau_fraction.
# If PHASE_PARAMETER="T0", provide T0_jd.
# If SCALE_PARAMETER="alpha_ast_mas", provide alpha_ast_mas.
# If SCALE_PARAMETER="K_mps", provide K_mps.
# If SCALE_PARAMETER="mass_mjup", provide mass_mjup.
PLANETS: List[Dict[str, float]] = [
    dict(
        name="b",
        P_day=653.9,
        e=0.479,
        K_mps=76.2,
        T0_jd=2453831.3,   # at ref epoch
        omega_deg=244.6,  # companion-relative argument of periastron
        cosi=0.9829353491,
        Omega_deg=123,
    ),
    dict(
        name="c",
        P_day=5387,
        e=0.196,
        K_mps=19.9,
        T0_jd=2452949,   # at ref epoch
        omega_deg=269,  # companion-relative argument of periastron
        cosi=0.988756381,
        Omega_deg=143.1,
    ),
]

# Single-measurement AL uncertainty. If USE_FIG_A1_AL_ERROR=True, sigma_AL is interpolated from G magnitude.
USE_FIG_A1_AL_ERROR = True
GAIA_ALSIG_CSV = r"/path/to/user_data/DR3_ALsig.csv"
FIGA1_GMAG_COL = "G magnitude"
FIGA1_SIGMA_COL = "Standard_deviation_of_AL_field_angle/mas"
TARGET_G_MAG = 9.332131
SIGMA_AL_MAS = 0.20  # Constant fallback AL uncertainty, mas, used only when USE_FIG_A1_AL_ERROR=False.

# ------------------------------------------------------------
# Gaia-like single-measurement AL noise model
# ------------------------------------------------------------
# Default option: use the blue curve digitized from Fig. A.1 of the Gaia EDR3 astrometric-solution paper.
# The curve represents the AL post-fit residual scatter / RSE as a function of G magnitude.
# This curve is used only as an empirical Gaia-like approximation to the scatter of an individual AL measurement.
# It is not a full Gaia error model and does not include CCD-, gate-, FoV-, colour-, or time-dependent systematics.
# ------------------------------------------------------------
# Behaviour outside the digitized Fig. A.1 magnitude range:
# The blue curve usually starts near G~5. For brighter targets, direct mathematical extrapolation is not recommended.
# Available BRIGHT_G_MODE options:
#   "inflate" : for G < G_min, use BRIGHT_G_FACTOR * sigma(G_min); recommended default.
#   "clamp"   : for G < G_min, use sigma(G_min).
#   "error"   : raise an error for G < G_min, forcing manual handling.
#   "extrapolate_log" : extrapolate in log(sigma); not recommended.
BRIGHT_G_MODE = "inflate"       # "inflate", "clamp", "error", "extrapolate_log"
BRIGHT_G_FACTOR = 2.0          # Multiplicative factor used by the "inflate" mode.
# Faint-end handling for G > G_max. The Fig. A.1 data usually extend to G~21, so this rarely matters.
FAINT_G_MODE = "clamp"          # "clamp", "error", "extrapolate_log"
AL_ERROR_FLOOR_MAS = 0.0        # Optional additional AL noise floor, in mas.

# Optional epoch-to-epoch G-band scatter, useful for a simple variability/window-photometry proxy.
RANDOMIZE_G_MAG_PER_TRANSIT = True
G_MAG_TRANSIT_SCATTER = 0.02  # Standard deviation of the epoch-level G magnitude perturbation.

# Optional outlier injection. Keep this False for a clean validation run.
INJECT_OUTLIERS = False
OUTLIER_FRACTION = 0.02   # Fraction of measurements flagged and perturbed as outliers when enabled.
OUTLIER_SIGMA_MAS = 5.0   # Additional outlier noise scale, in mas.

# True AL jitter used only in the truth JSON and truth parameter vector. The simulated observations add no extra jitter by default.
SIGMA_JIT_AL_TRUE_MAS = 0.0


# =============================================================================
# Constants and column names
# =============================================================================

PI = math.pi
TWOPI = 2.0 * math.pi
DEG2RAD = math.pi / 180.0
MJUP_TO_MSUN = 0.000954588
AU_M = 1.495978707e11

GOST_REQUIRED_COLUMNS = {
    "target": "Target",
    "ra_rad": "ra[rad]",
    "dec_rad": "dec[rad]",
    "obs_time_gaia_utc": "ObservationTimeAtGaia[UTC]",
    "ccd_row": "CcdRow[1-7]",
    "zeta_field_angle_rad": "zetaFieldAngle[rad]",
    "scan_angle_rad": "scanAngle[rad]",
    "fov": "Fov[FovP=preceding/FovF=following]",
    "parallax_factor_al": "parallaxFactorAlongScan",
    "parallax_factor_ac": "parallaxFactorAcrossScan",
    "obs_time_tcb": "ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]",
}

GOST_OPTIONAL_COLUMNS = {
    "ra_hms": "ra[h:m:s]",
    "dec_dms": "dec[d:m:s]",
}


# =============================================================================
# Basic utilities
# =============================================================================

def jyear_to_jd_tcb(jyear: float) -> float:
    return 2451545.0 + (float(jyear) - 2000.0) * 365.25


def get_t_ref_jd_tcb() -> float:
    if USE_CUSTOM_T_REF_JD_TCB:
        return float(CUSTOM_T_REF_JD_TCB)
    return jyear_to_jd_tcb(float(REF_EPOCH_JYEAR_TCB))


def wrap_to_2pi(x: float) -> float:
    y = math.fmod(float(x), TWOPI)
    if y < 0.0:
        y += TWOPI
    return y


def frac01(x: float) -> float:
    return float(x - math.floor(x))


def kepler_E(M: float, e: float) -> float:
    M = wrap_to_2pi(M)
    E = M if e < 0.8 else PI
    for _ in range(100):
        f = E - e * math.sin(E) - M
        fp = 1.0 - e * math.cos(E)
        if (not math.isfinite(f)) or (not math.isfinite(fp)) or abs(fp) < 1.0e-15:
            return float("nan")
        dE = -f / fp
        E += dE
        if abs(dE) < 1.0e-13:
            break
    return float(E)


def normalize_colname(name: str) -> str:
    return "".join(ch.lower() for ch in str(name).strip() if ch.isalnum())


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = [str(c).strip() for c in df.columns]
    if len(set(new_cols)) != len(new_cols):
        raise ValueError(f"Duplicate column names after stripping whitespace: {new_cols}")
    df.columns = new_cols
    return df


def get_exact_or_normalized_column(df: pd.DataFrame, expected_name: str) -> str:
    expected = expected_name.strip()
    if expected in df.columns:
        return expected
    expected_norm = normalize_colname(expected)
    matches = [c for c in df.columns if normalize_colname(c) == expected_norm]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise KeyError(f"Expected column {expected_name!r} matches multiple columns: {matches}")
    raise KeyError(f"Required GOST column {expected_name!r} was not found. Available columns: {list(df.columns)}")


def fov_to_code(value) -> int:
    s = str(value).strip().lower()
    if ("fovp" in s) or ("preced" in s) or (s == "p"):
        return -1
    if ("fovf" in s) or ("follow" in s) or (s == "f"):
        return +1
    return 0


def safe_sini_from_cosi(cosi: float) -> float:
    s2 = 1.0 - float(cosi) * float(cosi)
    if s2 <= 0.0:
        return 1.0e-12
    return math.sqrt(s2)


def alpha_star_mas_from_K(K_mps: float, period_days: float, ecc: float, cosi: float, plx_mas: float) -> float:
    P_sec = float(period_days) * 86400.0
    sini = safe_sini_from_cosi(float(cosi))
    fac = math.sqrt(max(0.0, 1.0 - float(ecc) * float(ecc)))
    a_star_m = float(K_mps) * P_sec * fac / (TWOPI * sini)
    a_star_au = a_star_m / AU_M
    return float(a_star_au * float(plx_mas))


def alpha_star_mas_from_mass(mp_mjup: float, period_days: float, mstar_msun: float, plx_mas: float) -> float:
    mp_msun = float(mp_mjup) * MJUP_TO_MSUN
    mtot = float(mstar_msun) + mp_msun
    period_yr = float(period_days) / 365.25
    a_rel_au = (period_yr * period_yr * mtot) ** (1.0 / 3.0)
    a_star_au = a_rel_au * mp_msun / mtot
    return float(a_star_au * float(plx_mas))


# =============================================================================
# Error model
# =============================================================================

def read_figA1_al_sigma_curve(csv_path: str,
                              g_col: str = FIGA1_GMAG_COL,
                              sigma_col: str = FIGA1_SIGMA_COL) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"AL uncertainty-curve file was not found: {csv_path}\n"
            "Set USE_FIG_A1_AL_ERROR=False if this file should not be used."
        )

    tab = pd.read_csv(path, sep=None, engine="python")
    tab = canonicalize_columns(tab)

    try:
        g_name = get_exact_or_normalized_column(tab, g_col)
    except KeyError:
        if tab.shape[1] >= 2:
            g_name = tab.columns[0]
        else:
            raise

    try:
        sig_name = get_exact_or_normalized_column(tab, sigma_col)
    except KeyError:
        if tab.shape[1] >= 2:
            sig_name = tab.columns[1]
        else:
            raise

    g = pd.to_numeric(tab[g_name], errors="coerce").to_numpy(dtype=float)
    sig = pd.to_numeric(tab[sig_name], errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(g) & np.isfinite(sig) & (sig > 0.0)
    if int(ok.sum()) < 2:
        raise ValueError("The AL uncertainty curve contains fewer than two valid points.")

    g = g[ok]
    sig = sig[ok]
    idx = np.argsort(g)
    g = g[idx]
    sig = sig[idx]

    unique_g = []
    unique_sig = []
    for val in np.unique(g):
        m = g == val
        unique_g.append(float(val))
        unique_sig.append(float(np.median(sig[m])))

    g_grid = np.asarray(unique_g, dtype=float)
    sigma_grid = np.asarray(unique_sig, dtype=float)
    if g_grid.size < 2 or not np.all(np.diff(g_grid) > 0.0):
        raise ValueError("The G-magnitude grid of the AL uncertainty curve is invalid.")
    return g_grid, sigma_grid


def interp_log_sigma_including_extrapolation(G: float, g_grid: np.ndarray, sigma_grid: np.ndarray) -> float:
    G = float(G)
    log_sig = np.log(np.asarray(sigma_grid, dtype=float))
    g = np.asarray(g_grid, dtype=float)

    if G <= g[0]:
        slope = (log_sig[1] - log_sig[0]) / (g[1] - g[0])
        return float(math.exp(log_sig[0] + slope * (G - g[0])))
    if G >= g[-1]:
        slope = (log_sig[-1] - log_sig[-2]) / (g[-1] - g[-2])
        return float(math.exp(log_sig[-1] + slope * (G - g[-1])))
    return float(math.exp(np.interp(G, g, log_sig)))


def sigma_al_from_figA1_gmag(G: float,
                             g_grid: np.ndarray,
                             sigma_grid: np.ndarray,
                             bright_mode: str = BRIGHT_G_MODE,
                             bright_factor: float = BRIGHT_G_FACTOR,
                             faint_mode: str = FAINT_G_MODE,
                             floor_mas: float = AL_ERROR_FLOOR_MAS) -> Tuple[float, str]:
    G = float(G)
    g_min = float(g_grid[0])
    g_max = float(g_grid[-1])
    bright_mode = str(bright_mode).strip().lower()
    faint_mode = str(faint_mode).strip().lower()

    if not math.isfinite(G):
        raise ValueError(f"TARGET_G_MAG is not finite: {G}")

    if G < g_min:
        if bright_mode == "clamp":
            sigma = float(sigma_grid[0])
            source = f"bright_clamp_to_Gmin_{g_min:.3f}"
        elif bright_mode == "inflate":
            sigma = float(bright_factor) * float(sigma_grid[0])
            source = f"bright_inflate_{bright_factor:.3g}_times_sigma_Gmin_{g_min:.3f}"
        elif bright_mode == "error":
            raise ValueError(f"Target G={G:.3f} is below the lower bound of the uncertainty curve, G_min={g_min:.3f}.")
        elif bright_mode == "extrapolate_log":
            sigma = interp_log_sigma_including_extrapolation(G, g_grid, sigma_grid)
            source = "bright_log_extrapolated"
        else:
            raise ValueError(f"Unknown BRIGHT_G_MODE={bright_mode!r}")
    elif G > g_max:
        if faint_mode == "clamp":
            sigma = float(sigma_grid[-1])
            source = f"faint_clamp_to_Gmax_{g_max:.3f}"
        elif faint_mode == "error":
            raise ValueError(f"Target G={G:.3f} is above the upper bound of the uncertainty curve, G_max={g_max:.3f}.")
        elif faint_mode == "extrapolate_log":
            sigma = interp_log_sigma_including_extrapolation(G, g_grid, sigma_grid)
            source = "faint_log_extrapolated"
        else:
            raise ValueError(f"Unknown FAINT_G_MODE={faint_mode!r}")
    else:
        sigma = float(math.exp(np.interp(G, g_grid, np.log(sigma_grid))))
        source = "log_interpolated"

    floor_mas = float(floor_mas)
    if floor_mas < 0.0:
        raise ValueError("AL_ERROR_FLOOR_MAS must not be negative.")
    if floor_mas > 0.0:
        sigma = math.sqrt(sigma * sigma + floor_mas * floor_mas)
        source += f"_plus_floor_{floor_mas:.4g}mas"

    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"Invalid AL sigma was computed: {sigma} mas.")
    return float(sigma), source


def make_sigma_al_array(n_al: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if n_al <= 0:
        raise ValueError("N_al must be positive.")

    if not USE_FIG_A1_AL_ERROR:
        sigma = float(SIGMA_AL_MAS)
        if sigma <= 0.0 or not math.isfinite(sigma):
            raise ValueError(f"Invalid SIGMA_AL_MAS: {SIGMA_AL_MAS}")
        sigma_al = np.full(n_al, sigma, dtype=float)
        g_each = np.full(n_al, np.nan, dtype=float)
        info = dict(mode="constant_sigma", sigma_al_mas=sigma)
        return sigma_al, g_each, info

    g_grid, sigma_grid = read_figA1_al_sigma_curve(GAIA_ALSIG_CSV)

    if RANDOMIZE_G_MAG_PER_TRANSIT:
        if float(G_MAG_TRANSIT_SCATTER) < 0.0:
            raise ValueError("G_MAG_TRANSIT_SCATTER must not be negative.")
        g_each = rng.normal(float(TARGET_G_MAG), float(G_MAG_TRANSIT_SCATTER), size=n_al)
    else:
        g_each = np.full(n_al, float(TARGET_G_MAG), dtype=float)

    sigma_vals = []
    labels = []
    for gval in g_each:
        sig, label = sigma_al_from_figA1_gmag(
            float(gval),
            g_grid,
            sigma_grid,
            bright_mode=BRIGHT_G_MODE,
            bright_factor=BRIGHT_G_FACTOR,
            faint_mode=FAINT_G_MODE,
            floor_mas=AL_ERROR_FLOOR_MAS,
        )
        sigma_vals.append(sig)
        labels.append(label)

    sigma_al = np.asarray(sigma_vals, dtype=float)
    info = dict(
        mode="FigA1_blue_curve_sigma",
        csv_path=str(GAIA_ALSIG_CSV),
        target_g_mag=float(TARGET_G_MAG),
        randomize_g_mag_per_transit=bool(RANDOMIZE_G_MAG_PER_TRANSIT),
        g_mag_transit_scatter=float(G_MAG_TRANSIT_SCATTER),
        sigma_al_median_mas=float(np.median(sigma_al)),
        sigma_al_min_mas=float(np.min(sigma_al)),
        sigma_al_max_mas=float(np.max(sigma_al)),
        sigma_source_labels=sorted(set(labels)),
    )
    return sigma_al, g_each, info


# =============================================================================
# GOST input
# =============================================================================

def read_gost_exact_columns(gost_csv: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(gost_csv, sep=None, engine="python")
    df = canonicalize_columns(df)

    matched: Dict[str, str] = {}
    for key, expected in GOST_REQUIRED_COLUMNS.items():
        matched[key] = get_exact_or_normalized_column(df, expected)

    optional_matched: Dict[str, str] = {}
    for key, expected in GOST_OPTIONAL_COLUMNS.items():
        try:
            optional_matched[key] = get_exact_or_normalized_column(df, expected)
        except KeyError:
            pass

    out = pd.DataFrame()
    out["target"] = df[matched["target"]].astype(str)
    out["ra_rad"] = pd.to_numeric(df[matched["ra_rad"]], errors="raise")
    out["dec_rad"] = pd.to_numeric(df[matched["dec_rad"]], errors="raise")
    out["ObservationTimeAtGaia_UTC"] = df[matched["obs_time_gaia_utc"]].astype(str)
    out["obs_time_tcb"] = pd.to_numeric(df[matched["obs_time_tcb"]], errors="raise")
    out["ccd_row"] = pd.to_numeric(df[matched["ccd_row"]], errors="raise").astype(int)
    out["zetaFieldAngle_rad"] = pd.to_numeric(df[matched["zeta_field_angle_rad"]], errors="raise")
    out["scanAngle_rad"] = pd.to_numeric(df[matched["scan_angle_rad"]], errors="raise")
    out["parallaxFactorAlongScan"] = pd.to_numeric(df[matched["parallax_factor_al"]], errors="raise")
    out["parallaxFactorAcrossScan"] = pd.to_numeric(df[matched["parallax_factor_ac"]], errors="raise")
    out["fov_label"] = df[matched["fov"]].astype(str).str.strip()
    out["fov_code"] = out["fov_label"].map(fov_to_code).astype(int)

    if "ra_hms" in optional_matched:
        out["ra_hms"] = df[optional_matched["ra_hms"]].astype(str)
    if "dec_dms" in optional_matched:
        out["dec_dms"] = df[optional_matched["dec_dms"]].astype(str)

    if not np.all(np.isfinite(out["obs_time_tcb"].to_numpy(dtype=float))):
        raise ValueError("obs_time_tcb contains non-finite values.")
    if not np.all(np.isfinite(out["scanAngle_rad"].to_numpy(dtype=float))):
        raise ValueError("scanAngle_rad contains non-finite values.")
    if not np.all(np.isfinite(out["parallaxFactorAlongScan"].to_numpy(dtype=float))):
        raise ValueError("parallaxFactorAlongScan contains non-finite values.")

    bad_rows = sorted(set(out.loc[~out["ccd_row"].between(1, 7), "ccd_row"].tolist()))
    if bad_rows:
        raise ValueError(f"CcdRow[1-7] contains values outside 1--7: {bad_rows}")

    out = out.sort_values("obs_time_tcb").reset_index(drop=True)
    out["synthetic_transit_id"] = np.arange(1, len(out) + 1, dtype=np.int64)

    matched.update(optional_matched)
    return out, matched


# =============================================================================
# Orbit model: keep the same phase and projection convention as AL_user_logll.c
# =============================================================================

def tau_from_planet_input(planet: Dict[str, float], t_ref_jd: float) -> float:
    mode = str(PHASE_PARAMETER).strip().lower()
    P_day = float(planet["P_day"])
    if mode == "tau":
        if "tau_fraction" not in planet:
            raise KeyError(f"Planet {planet.get('name', '')} is missing tau_fraction.")
        return frac01(float(planet["tau_fraction"]))
    if mode == "t0":
        if "T0_jd" not in planet:
            raise KeyError(f"Planet {planet.get('name', '')} is missing T0_jd.")
        return frac01((float(t_ref_jd) - float(planet["T0_jd"])) / P_day)
    raise ValueError("PHASE_PARAMETER must be either 'tau' or 'T0'.")


def alpha_from_planet_input(planet: Dict[str, float]) -> float:
    mode = str(SCALE_PARAMETER).strip().lower()
    P_day = float(planet["P_day"])
    e = float(planet["e"])
    cosi = float(planet["cosi"])
    plx_mas = float(STAR["parallax_mas"])

    if mode == "alpha_ast_mas":
        if "alpha_ast_mas" not in planet:
            raise KeyError(f"Planet {planet.get('name', '')} is missing alpha_ast_mas.")
        alpha = float(planet["alpha_ast_mas"])
    elif mode == "k_mps":
        if "K_mps" not in planet:
            raise KeyError(f"Planet {planet.get('name', '')} is missing K_mps.")
        alpha = alpha_star_mas_from_K(float(planet["K_mps"]), P_day, e, cosi, plx_mas)
    elif mode == "mass_mjup":
        if "mass_mjup" not in planet:
            raise KeyError(f"Planet {planet.get('name', '')} is missing mass_mjup.")
        alpha = alpha_star_mas_from_mass(float(planet["mass_mjup"]), P_day, float(MSTAR_MSUN), plx_mas)
    else:
        raise ValueError("SCALE_PARAMETER must be one of 'alpha_ast_mas', 'K_mps', or 'mass_mjup'.")

    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"Planet {planet.get('name', '')} has an invalid alpha_ast_mas: {alpha}")
    return float(alpha)


def validate_planet(planet: Dict[str, float]) -> None:
    name = planet.get("name", "")
    P_day = float(planet["P_day"])
    e = float(planet["e"])
    cosi = float(planet["cosi"])
    if not (P_day > 0.0 and math.isfinite(P_day)):
        raise ValueError(f"Planet {name} has an invalid P_day.")
    if not (0.0 <= e < 1.0 and math.isfinite(e)):
        raise ValueError(f"Planet {name} has an invalid eccentricity e.")
    if not (-1.0 <= cosi <= 1.0 and math.isfinite(cosi)):
        raise ValueError(f"Planet {name} has an invalid cosi.")


def calc_M_tau(t_jd: float, t_ref_jd: float, period_days: float, tau_fraction: float) -> float:
    return wrap_to_2pi(TWOPI * float(tau_fraction) + TWOPI * (float(t_jd) - float(t_ref_jd)) / float(period_days))


def thiele_innes(rho_mas: float,
                 omega_deg: float,
                 Omega_deg: float,
                 cosi: float) -> Tuple[float, float, float, float]:
    omega = float(omega_deg) * DEG2RAD
    Omega = float(Omega_deg) * DEG2RAD
    cO, sO = math.cos(Omega), math.sin(Omega)
    cw, sw = math.cos(omega), math.sin(omega)
    ci = float(cosi)

    A = rho_mas * (cw * cO - sw * sO * ci)
    B = rho_mas * (cw * sO + sw * cO * ci)
    F = rho_mas * (-sw * cO - cw * sO * ci)
    G = rho_mas * (-sw * sO + cw * cO * ci)
    return float(A), float(B), float(F), float(G)


def planet_reflex_offsets_mas(t_jd: float,
                              t_ref_jd: float,
                              planet: Dict[str, float],
                              tau_fraction: float,
                              alpha_ast_mas: float) -> Tuple[float, float]:
    M = calc_M_tau(t_jd, t_ref_jd, float(planet["P_day"]), tau_fraction)
    E = kepler_E(M, float(planet["e"]))
    if not math.isfinite(E):
        raise RuntimeError(f"Kepler equation failed to converge: planet={planet.get('name', '')}, t={t_jd}")

    e = float(planet["e"])
    X = math.cos(E) - e
    Y = math.sqrt(max(0.0, 1.0 - e * e)) * math.sin(E)

    A, B, F, G = thiele_innes(
        rho_mas=-float(alpha_ast_mas),
        omega_deg=float(planet["omega_deg"]),
        Omega_deg=float(planet["Omega_deg"]),
        cosi=float(planet["cosi"]),
    )

    dalpha_star_mas = B * X + G * Y
    ddelta_mas = A * X + F * Y
    return float(dalpha_star_mas), float(ddelta_mas)


def prepared_planets(t_ref_jd: float) -> List[Dict[str, float]]:
    out = []
    for planet in PLANETS:
        validate_planet(planet)
        tau = tau_from_planet_input(planet, t_ref_jd)
        alpha = alpha_from_planet_input(planet)
        item = dict(planet)
        item["tau_fraction_for_C"] = float(tau)
        item["alpha_ast_mas_for_C"] = float(alpha)
        out.append(item)
    return out


def truth_parameter_vector_for_c(prepared: List[Dict[str, float]]) -> List[float]:
    vec: List[float] = []
    for pl in prepared:
        vec.extend([
            float(pl["P_day"]),
            float(pl["e"]),
            float(pl["alpha_ast_mas_for_C"]),
            float(pl["tau_fraction_for_C"]),
            float(pl["omega_deg"]),
            float(pl["cosi"]),
            float(pl["Omega_deg"]),
        ])
    vec.append(float(SIGMA_JIT_AL_TRUE_MAS))
    return vec


# =============================================================================
# Main routine
# =============================================================================

def simulate() -> None:
    rng = np.random.default_rng(int(RANDOM_SEED))

    t_ref = get_t_ref_jd_tcb()
    planets = prepared_planets(t_ref)

    gost, matched_cols = read_gost_exact_columns(GOST_CSV)
    n_al = len(gost)
    if n_al <= 0:
        raise RuntimeError("The GOST file contains no valid observing events.")

    times = gost["obs_time_tcb"].to_numpy(dtype=float)
    scan_angle = gost["scanAngle_rad"].to_numpy(dtype=float)
    sin_psi = np.sin(scan_angle)
    cos_psi = np.cos(scan_angle)
    pal = gost["parallaxFactorAlongScan"].to_numpy(dtype=float)
    dt_yr = (times - t_ref) / 365.25

    dra_5p = float(STAR["dalpha0_star_mas"]) + float(STAR["pmra_star_masyr"]) * dt_yr
    ddec_5p = float(STAR["ddelta0_mas"]) + float(STAR["pmdec_masyr"]) * dt_yr

    dra_orb_total = np.zeros(n_al, dtype=float)
    ddec_orb_total = np.zeros(n_al, dtype=float)
    per_planet_debug: Dict[str, Dict[str, np.ndarray]] = {}

    for pl in planets:
        dra_list = []
        ddec_list = []
        for t in times:
            da, dd = planet_reflex_offsets_mas(
                float(t),
                t_ref,
                pl,
                float(pl["tau_fraction_for_C"]),
                float(pl["alpha_ast_mas_for_C"]),
            )
            dra_list.append(da)
            ddec_list.append(dd)
        dra_arr = np.asarray(dra_list, dtype=float)
        ddec_arr = np.asarray(ddec_list, dtype=float)
        dra_orb_total += dra_arr
        ddec_orb_total += ddec_arr
        name = str(pl.get("name", f"planet{len(per_planet_debug) + 1}"))
        per_planet_debug[name] = {"dra": dra_arr, "ddec": ddec_arr}

    al_5p_true = dra_5p * sin_psi + ddec_5p * cos_psi + float(STAR["parallax_mas"]) * pal
    al_orbit_true = dra_orb_total * sin_psi + ddec_orb_total * cos_psi
    al_true = al_5p_true + al_orbit_true

    sigma_al, g_mag_for_error_model, error_model_info = make_sigma_al_array(n_al, rng)
    noise = rng.normal(0.0, sigma_al)
    outlier_flag = np.zeros(n_al, dtype=int)
    outlier_noise = np.zeros(n_al, dtype=float)

    if INJECT_OUTLIERS and float(OUTLIER_FRACTION) > 0.0:
        mask = rng.random(n_al) < float(OUTLIER_FRACTION)
        outlier_flag[mask] = 1
        outlier_noise[mask] = rng.normal(0.0, float(OUTLIER_SIGMA_MAS), size=int(mask.sum()))

    al_obs = al_true + noise + outlier_noise

    df_out = pd.DataFrame({
        "target_name": TARGET_NAME,
        "gost_target": gost["target"].to_numpy(),
        "synthetic_transit_id": gost["synthetic_transit_id"].to_numpy(),
        "ccd_row": gost["ccd_row"].to_numpy(dtype=int),
        "fov_label": gost["fov_label"].to_numpy(),
        "fov_code": gost["fov_code"].to_numpy(dtype=int),
        "obs_time_tcb": times,
        "obs_time_tcb_jyear": 2000.0 + (times - 2451545.0) / 365.25,
        "dt_yr_from_ref_epoch": dt_yr,
        "ObservationTimeAtGaia_UTC": gost["ObservationTimeAtGaia_UTC"].to_numpy(),
        "ra_rad_from_gost": gost["ra_rad"].to_numpy(dtype=float),
        "dec_rad_from_gost": gost["dec_rad"].to_numpy(dtype=float),
        "zetaFieldAngle_rad": gost["zetaFieldAngle_rad"].to_numpy(dtype=float),
        "scanAngle_rad": scan_angle,
        "scanAngle_deg": np.rad2deg(scan_angle),
        "sin_psi": sin_psi,
        "cos_psi": cos_psi,
        "parallaxFactorAlongScan": pal,
        "parallaxFactorAcrossScan": gost["parallaxFactorAcrossScan"].to_numpy(dtype=float),
        "centroid_pos_al_mas": al_obs,
        "centroid_pos_error_al_mas": sigma_al,
        "g_mag_for_al_error_model": g_mag_for_error_model,
        "outlier_flag": outlier_flag,
        "al_true_mas": al_true,
        "al_5p_true_mas": al_5p_true,
        "al_orbit_true_mas": al_orbit_true,
        "noise_mas": noise,
        "outlier_noise_mas": outlier_noise,
        "dalpha_5p_mas": dra_5p,
        "ddelta_5p_mas": ddec_5p,
        "dalpha_star_orbit_total_mas": dra_orb_total,
        "ddelta_orbit_total_mas": ddec_orb_total,
    })

    for name, comp in per_planet_debug.items():
        df_out[f"dalpha_star_orbit_{name}_mas"] = comp["dra"]
        df_out[f"ddelta_orbit_{name}_mas"] = comp["ddec"]

    if "ra_hms" in gost.columns:
        df_out["ra_hms_from_gost"] = gost["ra_hms"].to_numpy()
    if "dec_dms" in gost.columns:
        df_out["dec_dms_from_gost"] = gost["dec_dms"].to_numpy()

    ncol = 11
    mat = np.zeros((1 + n_al, ncol), dtype=float)
    t_ref_floor = math.floor(t_ref)
    t_ref_frac = t_ref - t_ref_floor

    mat[0, 0] = 1.0
    mat[0, 1] = float(n_al)
    mat[0, 2] = float(t_ref_floor)
    mat[0, 3] = float(t_ref_frac)

    for i in range(n_al):
        mat[1 + i, 0] = times[i]
        mat[1 + i, 1] = al_obs[i]
        mat[1 + i, 2] = sigma_al[i]
        mat[1 + i, 3] = scan_angle[i]
        mat[1 + i, 4] = sin_psi[i]
        mat[1 + i, 5] = cos_psi[i]
        mat[1 + i, 6] = pal[i]
        mat[1 + i, 7] = float(outlier_flag[i])
        mat[1 + i, 8] = float(gost.loc[i, "synthetic_transit_id"])
        mat[1 + i, 9] = float(gost.loc[i, "ccd_row"])
        mat[1 + i, 10] = float(gost.loc[i, "fov_code"])

    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    debug_csv_path = outdir / f"{TARGET_NAME}_gaia_al_raw_debug_j2016.csv"
    matrix_txt_path = outdir / f"{TARGET_NAME}_gaia_al_raw_matrix_j2016.txt"
    matrix_dat_path = outdir / f"{TARGET_NAME}_gaia_al_raw_matrix_j2016.dat"
    matrix_csv_path = outdir / f"{TARGET_NAME}_gaia_al_raw_matrix_j2016.csv"
    truth_path = outdir / f"{TARGET_NAME}_truth_j2016.json"

    # The debug CSV keeps diagnostic columns; the matrix files are the compact fitting inputs.
    df_out.to_csv(debug_csv_path, index=False, encoding="utf-8-sig")
    np.savetxt(matrix_txt_path, mat, fmt="%.18e")
    np.savetxt(matrix_dat_path, mat, fmt="%.18e")
    np.savetxt(matrix_csv_path, mat, fmt="%.18e", delimiter=",")

    truth_vec = truth_parameter_vector_for_c(planets)
    truth = {
        "target_name": TARGET_NAME,
        "data_type": "synthetic local-plane along-scan astrometry for AL_user_logll.c",
        "reference_epoch_jyear_tcb": None if USE_CUSTOM_T_REF_JD_TCB else float(REF_EPOCH_JYEAR_TCB),
        "t_ref_jd_tcb": float(t_ref),
        "t_ref_row0_split": [float(t_ref_floor), float(t_ref_frac)],
        "N_al": int(n_al),
        "N_planet_in_simulator": int(len(planets)),
        "phase_parameter": str(PHASE_PARAMETER),
        "scale_parameter": str(SCALE_PARAMETER),
        "star_parameters": STAR,
        "planet_parameters_user_input": PLANETS,
        "planet_parameters_for_C": planets,
        "sigma_jit_AL_true_mas": float(SIGMA_JIT_AL_TRUE_MAS),
        "parameter_vector_for_AL_user_logll": truth_vec,
        "parameter_order_for_AL_user_logll": [
            "P_day", "e", "alpha_ast_mas", "tau_fraction", "omega_rel_deg", "cosi", "Omega_deg",
            "... repeated for each planet ...",
            "sigma_jit_AL_mas",
        ],
        "al_error_model": error_model_info,
        "gost_column_mapping_used": matched_cols,
        "matrix_files": {
            "txt": str(matrix_txt_path),
            "dat": str(matrix_dat_path),
            "csv": str(matrix_csv_path),
        },
        "matrix_format": {
            "row0": ["AL_start_row", "N_al", "floor(t_ref_jd_tcb)", "fractional_t_ref_jd_tcb"],
            "data_row": [
                "obs_time_tcb",
                "centroid_pos_al_mas",
                "centroid_pos_error_al_mas",
                "scanAngle_rad",
                "sin_psi",
                "cos_psi",
                "parallaxFactorAlongScan",
                "outlier_flag",
                "synthetic_transit_id",
                "ccd_row",
                "fov_code",
            ],
        },
    }
    with open(truth_path, "w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Gaia-like raw AL simulation finished")
    print(f"target                  = {TARGET_NAME}")
    print(f"N_al                    = {n_al}")
    print(f"N_planet                = {len(planets)}")
    print(f"t_ref_jd_tcb            = {t_ref:.12f}")
    print(f"row0 t_ref split        = {t_ref_floor:.0f} + {t_ref_frac:.12f}")
    print(f"phase parameter         = {PHASE_PARAMETER}")
    print(f"scale parameter         = {SCALE_PARAMETER}")
    print(f"AL error model          = {error_model_info.get('mode')}")
    print(f"sigma_AL median/min/max = {np.median(sigma_al):.6f} / {np.min(sigma_al):.6f} / {np.max(sigma_al):.6f} mas")
    print("Truth parameter vector for AL_user_logll.c:")
    for idx, val in enumerate(truth_vec):
        print(f"  para{idx:02d} = {val:.16e}")
    print(f"Saved debug CSV         = {debug_csv_path}")
    print(f"Saved matrix TXT        = {matrix_txt_path}")
    print(f"Saved matrix DAT        = {matrix_dat_path}")
    print(f"Saved matrix CSV        = {matrix_csv_path}")
    print(f"Saved truth JSON        = {truth_path}")
    print("=" * 80)


if __name__ == "__main__":
    simulate()

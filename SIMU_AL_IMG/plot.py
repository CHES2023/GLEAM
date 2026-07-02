#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AL + IMG profile-4p mass-mode joint-fit plotting script
=======================================================

Direct usage:
    python3 plot_al_img_profile4p_modified.py

Default inputs:
    DATA_MATRIX = "input.dat"
    CHAIN_FILE  = "chains/chain7.dat"
    INPUT_INI   = "input.ini"

Matched C likelihood convention
-------------------------------
This script is matched to the uploaded AL+IMG profile-4p likelihood:

row0 of input.dat:
    col0  = AL_start_row, usually 1
    col1  = IMG_start_row = 1 + N_AL
    col2  = N_img_epoch
    col3  = ep1a = floor(t_ref_jd)
    col4  = ep1b = t_ref_jd - ep1a
    col5... = Mi per IMG epoch, each Mi = 0..NPLANET

All IMG detections are treated as confirmed companion-relative astrometry.
Within one IMG epoch, two detections cannot be assigned to the same companion.

AL rows:
    col0  = obs_time_tcb_jd
    col1  = centroid_pos_al_mas
    col2  = centroid_pos_error_al_mas
    col3  = scanAngle_rad
    col4  = sin_psi
    col5  = cos_psi
    col6  = parallaxFactorAlongScan
    col7  = outlier_flag, 0 used, 1 rejected
    col8  = transit_id, optional diagnostic only
    col9  = ccd_row, optional diagnostic only
    col10 = fov_code, optional diagnostic only

IMG rows:
    col0 = t_jd
    col1 = delta_alpha_star_mas, East positive
    col2 = delta_delta_mas, North positive
    col3 = sigma_delta_alpha_star_mas
    col4 = sigma_delta_delta_mas

Parameter layout for NPLANET=2, N_parm=17:
    P1, e1, Mp1_Mjup, tau1_frac, omega1_planet_deg, cosi1, Omega1_deg,
    P2, e2, Mp2_Mjup, tau2_frac, omega2_planet_deg, cosi2, Omega2_deg,
    parallax_mas, sigma_jit_AL_mas, sigma_jit_IMG_mas

Geometry convention:
    omega_planet_deg is the planet/companion relative-to-star argument of periastron.
    IMG uses the relative planet vector directly.
    AL uses stellar reflex = -Mp/(Mstar+Mp) * relative planet vector.
    tau_frac sets the mean anomaly at t_ref: M(t_ref) = 2*pi*tau_frac.

Main output
-----------
Only one PNG figure is written by default:
    05_al_img_orbits_with_al_residuals_wide.png

This wide paper-style figure contains the AL sky-plane reflex orbit,
the IMG relative-orbit plot, and the AL O-C residuals.
CSV/TXT diagnostics are still written to OUTDIR.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 0. User settings
# ============================================================
DATA_MATRIX = "input.dat"
CHAIN_FILE = "chains/chain7.dat"
INPUT_INI = "input.ini"
OUTDIR = "fit_plots_al_img"
COMBINED_FIGURE_NAME = "05_al_img_orbits_with_al_residuals_wide.png"

NPLANET = 2
LOGLIKE_COL = -3
PARAM_START_COL = 0
BURNIN_FRACTION = 0.80

# If INPUT_INI cannot be read, set this manually in grams, e.g. 2.62548e33.
MSTAR_IN_G_FALLBACK: Optional[float] = None

# Plot controls
ORBIT_NGRID = 1800
PHASE_NGRID = 800
PHASE_SMOOTH_WINDOW = 31
SHOW_AL_CONSTRAINT_LINES = True
LINE_HALF_LENGTH_MAS = 1.5
IMG_LIKELY_D2_THRESH = 9.0
USE_INTERNAL_CONFIG_BY_DEFAULT = True

# ============================================================
# 1. Constants
# ============================================================
MSUN_G = 1.98847e33
MJUP_G = 1.89813e30
MJUP_TO_MSUN = MJUP_G / MSUN_G
AU_M = 1.495978707e11
DAY_S = 86400.0
TWOPI = 2.0 * np.pi
DEG2RAD = np.pi / 180.0


def object_letter_label(k: int) -> str:
    """Return the paper-style object label for zero-based companion index k."""
    if k == 0:
        return "secondary star B"
    if k == 1:
        return r"planet $c$"
    return f"planet ${chr(98 + k)}$"


def object_orbit_label(k: int) -> str:
    """Return the compact orbit-legend label for zero-based companion index k."""
    if k == 0:
        return "secondary star B orbit"
    if k == 1:
        return r"planet $c$ orbit"
    return f"planet ${chr(98 + k)}$ orbit"


# ============================================================
# 2. Basic helpers
# ============================================================
def read_ini_value(path: str | Path, key: str) -> Optional[float]:
    path = Path(path)
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k.strip() == key:
            try:
                return float(v.strip().split(",")[0].split()[0])
            except Exception:
                return None
    return None


def get_mstar_in_g(input_ini: str | Path) -> float:
    v = read_ini_value(input_ini, "Mstar_in_g")
    if v is not None and np.isfinite(v) and v > 0:
        return float(v)
    v_msun = read_ini_value(input_ini, "Mstar_Msun")
    if v_msun is not None and np.isfinite(v_msun) and v_msun > 0:
        return float(v_msun) * MSUN_G
    if MSTAR_IN_G_FALLBACK is not None:
        return float(MSTAR_IN_G_FALLBACK)
    raise RuntimeError(
        "Cannot read stellar mass. Add 'Mstar_in_g: ...' to input.ini, "
        "or set MSTAR_IN_G_FALLBACK near the top of this script."
    )


def safe_sigma(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return np.where(np.asarray(x, dtype=float) > 0.0, np.asarray(x, dtype=float), floor)


def kepler_E(M: np.ndarray, e: float, max_iter: int = 100) -> np.ndarray:
    M = np.mod(np.asarray(M, dtype=float), TWOPI)
    E = np.where(e < 0.8, M, np.pi * np.ones_like(M))
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E = E + dE
        if np.nanmax(np.abs(dE)) < 1e-13:
            break
    return E


def thiele_innes(a_scale: float, omega_deg: float, Omega_deg: float, cosi: float) -> Tuple[float, float, float, float]:
    omega = omega_deg * DEG2RAD
    Omega = Omega_deg * DEG2RAD
    cO, sO = np.cos(Omega), np.sin(Omega)
    cw, sw = np.cos(omega), np.sin(omega)
    A = a_scale * (cw * cO - sw * sO * cosi)
    B = a_scale * (cw * sO + sw * cO * cosi)
    F = a_scale * (-sw * cO - cw * sO * cosi)
    G = a_scale * (-sw * sO + cw * cO * cosi)
    return float(A), float(B), float(F), float(G)


def a_rel_AU_from_period_mass(P_day: float, Mstar_Msun: float, Mp_Mjup: float) -> float:
    P_yr = P_day / 365.25
    Mp_Msun = Mp_Mjup * MJUP_TO_MSUN
    if not (P_yr > 0.0 and Mstar_Msun + Mp_Msun > 0.0):
        return np.nan
    return float(((Mstar_Msun + Mp_Msun) * P_yr * P_yr) ** (1.0 / 3.0))


def calc_K_mps(Mstar_Msun: float, Mp_Mjup: float, P_day: float, e: float, cosi: float) -> float:
    sini = math.sqrt(max(0.0, 1.0 - cosi * cosi))
    P_yr = P_day / 365.25
    Mp_Msun = Mp_Mjup * MJUP_TO_MSUN
    if sini <= 0.0 or e >= 1.0 or P_yr <= 0.0:
        return np.nan
    return float(
        28.4329 / math.sqrt(max(1e-300, 1.0 - e * e))
        * Mp_Mjup * sini * (Mstar_Msun + Mp_Msun) ** (-2.0 / 3.0) * P_yr ** (-1.0 / 3.0)
    )


# ============================================================
# 3. Input parsers
# ============================================================
def read_matrix(path: str | Path) -> np.ndarray:
    mat = np.loadtxt(path)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.shape[0] < 2:
        raise ValueError(f"Input matrix has too few rows: {path}")
    return mat


def parse_matrix(mat: np.ndarray) -> Dict:
    row0 = mat[0]
    iStart_AL = int(round(row0[0]))
    iStart_IMG = int(round(row0[1]))
    N_img_epoch = int(round(row0[2]))
    t_ref = float(row0[3] + row0[4])

    if iStart_AL < 1:
        iStart_AL = 1
    if iStart_IMG < iStart_AL:
        raise ValueError("IMG_start_row must be >= AL_start_row")
    if 5 + N_img_epoch > len(row0):
        raise ValueError("row0 does not contain enough Mi columns for N_img_epoch")

    Mi = np.array([int(round(row0[5 + i])) for i in range(N_img_epoch)], dtype=int)
    if np.any((Mi < 0) | (Mi > NPLANET)):
        raise ValueError(f"Each Mi must be 0..NPLANET, got {Mi.tolist()}")

    N_al = iStart_IMG - iStart_AL
    N_img_total = mat.shape[0] - iStart_IMG
    if Mi.sum() != N_img_total:
        raise ValueError(f"Mi.sum()={Mi.sum()} != IMG data rows={N_img_total}. Check row0 Mi values.")

    Eidx = np.zeros(N_img_epoch, dtype=int)
    for i in range(1, N_img_epoch):
        Eidx[i] = Eidx[i - 1] + Mi[i - 1]

    al = mat[iStart_AL:iStart_IMG]
    img = mat[iStart_IMG:]
    if N_al > 0 and al.shape[1] < 8:
        raise ValueError(f"AL rows require at least 8 columns, got {al.shape[1]}")
    if N_img_total > 0 and img.shape[1] < 5:
        raise ValueError(f"IMG rows require at least 5 columns, got {img.shape[1]}")

    return dict(
        row0=row0,
        iStart_AL=iStart_AL,
        iStart_IMG=iStart_IMG,
        N_img_epoch=N_img_epoch,
        t_ref=t_ref,
        # Kept only for backward-compatible diagnostics. The current model
        # treats all IMG detections as confirmed companion-relative astrometry.
        img_source_mode=1,
        nplanet_input=float("nan"),
        Mi=Mi,
        Eidx=Eidx,
        N_al=N_al,
        N_img_total=N_img_total,
        al=al,
        img=img,
    )

def make_al_dataframe(al: np.ndarray) -> pd.DataFrame:
    n = len(al)
    def col(j: int, default: float = np.nan) -> np.ndarray:
        if al.shape[1] > j:
            return al[:, j]
        return np.full(n, default, dtype=float)

    return pd.DataFrame({
        "obs_time_tcb": col(0),
        "centroid_pos_al_mas": col(1),
        "centroid_pos_error_al_mas": col(2),
        "scanAngle_rad": col(3),
        "sin_psi": col(4),
        "cos_psi": col(5),
        "parallaxFactorAlongScan": col(6),
        "outlier_flag": col(7, 0.0),
        "transit_id": col(8),
        "ccd_row": col(9),
        "fov_code": col(10),
    })


def read_chain(path: str | Path,
               npars: int,
               param_start_col: int,
               loglike_col: int,
               burnin_fraction: float) -> Dict:
    chain = np.loadtxt(path)
    if chain.ndim == 1:
        chain = chain.reshape(1, -1)
    if chain.shape[1] < param_start_col + npars:
        raise ValueError(
            f"Chain file has only {chain.shape[1]} columns, but {param_start_col + npars} parameter columns are required."
        )
    loglike = chain[:, loglike_col]
    finite_ll = np.isfinite(loglike)
    if not np.any(finite_ll):
        raise ValueError("The selected log-likelihood column contains no finite values.")

    # Best row is selected after burn-in, matching the previous plotting script convention.
    n = len(chain)
    i0 = int(max(0, min(n - 1, math.floor(np.clip(burnin_fraction, 0.0, 0.95) * n))))
    sub = chain[i0:]
    sub_loglike = sub[:, loglike_col]
    best_local = int(np.nanargmax(np.where(np.isfinite(sub_loglike), sub_loglike, -np.inf)))
    best_idx = i0 + best_local
    best_row = chain[best_idx]

    samples = chain[i0:, param_start_col:param_start_col + npars]
    samples = samples[np.all(np.isfinite(samples), axis=1)]
    if len(samples) == 0:
        raise ValueError("No finite posterior samples remain after burn-in.")

    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    return dict(
        chain=chain,
        best_row=best_row,
        best_idx=best_idx,
        best_loglike=float(loglike[best_idx]),
        samples=samples,
        p16=q16,
        p50=q50,
        p84=q84,
        burnin_start=i0,
    )


# ============================================================
# 4. Orbit and model functions
# ============================================================
def unpack_planets(params: np.ndarray, Mstar_Msun: float, nplanet: int) -> List[Dict]:
    planets: List[Dict] = []
    for k in range(nplanet):
        j = 7 * k
        P = float(params[j + 0])
        e = float(params[j + 1])
        Mp = float(params[j + 2])
        tau = float(params[j + 3])
        omega = float(params[j + 4])
        cosi = float(params[j + 5])
        Omega = float(params[j + 6])
        a_rel = a_rel_AU_from_period_mass(P, Mstar_Msun, Mp)
        Mp_Msun = Mp * MJUP_TO_MSUN
        fstar = Mp_Msun / (Mstar_Msun + Mp_Msun) if Mstar_Msun + Mp_Msun > 0 else np.nan
        A, B, F, G = thiele_innes(a_rel, omega, Omega, cosi)
        planets.append(dict(
            index=k + 1,
            P_day=P,
            e=e,
            Mp_Mjup=Mp,
            tau_frac=tau - math.floor(tau),
            omega_planet_deg=omega,
            omega_rel_deg=omega,  # backward-compatible alias for existing labels/tables
            cosi=cosi,
            Omega_deg=Omega,
            a_rel_AU=a_rel,
            f_star=fstar,
            A=A, B=B, F=F, G=G,
        ))
    return planets

def rel_offset_AU(p: Dict, t: np.ndarray, t_ref: float) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    M = TWOPI * p["tau_frac"] + TWOPI * (t - t_ref) / p["P_day"]
    E = kepler_E(M, p["e"])
    X = np.cos(E) - p["e"]
    Y = np.sqrt(max(0.0, 1.0 - p["e"] * p["e"])) * np.sin(E)
    # East-positive Δα* and north-positive Δδ, matching the C code convention.
    dra = p["B"] * X + p["G"] * Y
    ddec = p["A"] * X + p["F"] * Y
    return dra, ddec

def relative_offset_mas(p: Dict, t: np.ndarray, t_ref: float, plx_mas: float) -> Tuple[np.ndarray, np.ndarray]:
    dra_au, ddec_au = rel_offset_AU(p, t, t_ref)
    return dra_au * plx_mas, ddec_au * plx_mas


def stellar_reflex_offset_mas(p: Dict, t: np.ndarray, t_ref: float, plx_mas: float) -> Tuple[np.ndarray, np.ndarray]:
    dra_rel, ddec_rel = relative_offset_mas(p, t, t_ref, plx_mas)
    return -p["f_star"] * dra_rel, -p["f_star"] * ddec_rel


def compute_al_orbit_components(planets: List[Dict],
                                t: np.ndarray,
                                t_ref: float,
                                plx_mas: float,
                                sinpsi: np.ndarray,
                                cospsi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    dra_total = np.zeros_like(t, dtype=float)
    ddec_total = np.zeros_like(t, dtype=float)
    al_total = np.zeros_like(t, dtype=float)
    per_dra: List[np.ndarray] = []
    per_ddec: List[np.ndarray] = []
    per_al: List[np.ndarray] = []
    for p in planets:
        dra_k, ddec_k = stellar_reflex_offset_mas(p, t, t_ref, plx_mas)
        al_k = dra_k * sinpsi + ddec_k * cospsi
        dra_total += dra_k
        ddec_total += ddec_k
        al_total += al_k
        per_dra.append(dra_k)
        per_ddec.append(ddec_k)
        per_al.append(al_k)
    return dra_total, ddec_total, al_total, per_dra, per_ddec, per_al


def solve_profile4p_AL(df: pd.DataFrame,
                       planets: List[Dict],
                       t_ref: float,
                       plx_mas: float,
                       sigma_jit_AL: float) -> Tuple[np.ndarray, Dict[str, float]]:
    t = df["obs_time_tcb"].to_numpy(dtype=float)
    y = df["centroid_pos_al_mas"].to_numpy(dtype=float)
    sig = safe_sigma(df["centroid_pos_error_al_mas"].to_numpy(dtype=float))
    sinpsi = df["sin_psi"].to_numpy(dtype=float)
    cospsi = df["cos_psi"].to_numpy(dtype=float)
    pal = df["parallaxFactorAlongScan"].to_numpy(dtype=float)
    flag = df["outlier_flag"].to_numpy(dtype=float)

    dt = (t - t_ref) / 365.25
    _, _, al_orb, _, _, _ = compute_al_orbit_components(planets, t, t_ref, plx_mas, sinpsi, cospsi)
    sigma_eff = np.sqrt(sig * sig + sigma_jit_AL * sigma_jit_AL)
    used = (
        np.isfinite(t) & np.isfinite(y) & np.isfinite(sig) & (sig > 0.0)
        & np.isfinite(sinpsi) & np.isfinite(cospsi) & np.isfinite(pal)
        & np.isfinite(al_orb) & (np.rint(flag).astype(int) == 0)
    )
    if int(used.sum()) < 4:
        raise RuntimeError(f"AL good points < 4; got {int(used.sum())}. Cannot solve profile-4p.")

    # C likelihood profile-4p: parallax is sampled/fixed here, not solved linearly.
    yprime = y - plx_mas * pal - al_orb
    H = np.column_stack([sinpsi, cospsi, dt * sinpsi, dt * cospsi])
    Hw = H[used] / sigma_eff[used, None]
    yw = yprime[used] / sigma_eff[used]
    x4, residuals, rank, svals = np.linalg.lstsq(Hw, yw, rcond=None)
    info = {
        "n_used_for_profile4p": float(used.sum()),
        "rank_profile4p": float(rank),
        "cond_profile4p": float(svals[0] / svals[-1]) if len(svals) and svals[-1] > 0 else np.inf,
    }
    return x4, info


def evaluate_al_model(df: pd.DataFrame,
                      planets: List[Dict],
                      t_ref: float,
                      plx_mas: float,
                      sigma_jit_AL: float) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, float]]:
    x4, ls_info = solve_profile4p_AL(df, planets, t_ref, plx_mas, sigma_jit_AL)

    t = df["obs_time_tcb"].to_numpy(dtype=float)
    y = df["centroid_pos_al_mas"].to_numpy(dtype=float)
    sig = safe_sigma(df["centroid_pos_error_al_mas"].to_numpy(dtype=float))
    psi = df["scanAngle_rad"].to_numpy(dtype=float)
    sinpsi = df["sin_psi"].to_numpy(dtype=float)
    cospsi = df["cos_psi"].to_numpy(dtype=float)
    pal = df["parallaxFactorAlongScan"].to_numpy(dtype=float)
    flag = df["outlier_flag"].to_numpy(dtype=float)
    dt = (t - t_ref) / 365.25

    dra_orb, ddec_orb, al_orb, per_dra, per_ddec, per_al = compute_al_orbit_components(
        planets, t, t_ref, plx_mas, sinpsi, cospsi
    )
    al_single = x4[0] * sinpsi + x4[1] * cospsi + x4[2] * dt * sinpsi + x4[3] * dt * cospsi + plx_mas * pal
    al_model = al_single + al_orb
    sigma_eff = np.sqrt(sig * sig + sigma_jit_AL * sigma_jit_AL)
    resid = y - al_model
    used = (
        np.isfinite(t) & np.isfinite(y) & np.isfinite(sig) & (sig > 0.0)
        & np.isfinite(sinpsi) & np.isfinite(cospsi) & np.isfinite(pal)
        & np.isfinite(al_orb) & (np.rint(flag).astype(int) == 0)
    )

    out = df.copy()
    out["dt_yr"] = dt
    out["al_single_model_mas"] = al_single
    out["al_orbit_model_mas"] = al_orb
    out["al_model_mas"] = al_model
    out["al_obs_minus_single_mas"] = y - al_single
    out["al_residual_mas"] = resid
    out["al_residual_over_sigma"] = resid / sigma_eff
    out["sigma_eff_mas"] = sigma_eff
    out["dalpha_orbit_total_mas"] = dra_orb
    out["ddelta_orbit_total_mas"] = ddec_orb
    # AL pseudo-point: the closest sky-plane point along the measured scan direction.
    out["dalpha_pseudopoint_mas"] = dra_orb + resid * sinpsi
    out["ddelta_pseudopoint_mas"] = ddec_orb + resid * cospsi
    out["used_in_likelihood_like_C"] = used

    for k in range(len(planets)):
        pnum = k + 1
        out[f"al_orbit_model_p{pnum}_mas"] = per_al[k]
        out[f"dalpha_orbit_p{pnum}_mas"] = per_dra[k]
        out[f"ddelta_orbit_p{pnum}_mas"] = per_ddec[k]

    return out, x4, ls_info


# ============================================================
# 5. IMG diagnostics and assignments
# ============================================================
def compute_img_diagnostics(data: Dict,
                            planets: List[Dict],
                            plx_mas: float,
                            sigma_jit_IMG: float) -> Dict:
    img = data["img"]
    n = len(img)
    if n == 0:
        return dict(assignments=pd.DataFrame(), t=np.array([]), da=np.array([]), dd=np.array([]))

    t = img[:, 0].astype(float)
    da = img[:, 1].astype(float)
    dd = img[:, 2].astype(float)
    sda = safe_sigma(img[:, 3].astype(float))
    sdd = safe_sigma(img[:, 4].astype(float))
    se_da = np.sqrt(sda * sda + sigma_jit_IMG * sigma_jit_IMG)
    se_dd = np.sqrt(sdd * sdd + sigma_jit_IMG * sigma_jit_IMG)

    model_da: List[np.ndarray] = []
    model_dd: List[np.ndarray] = []
    d2: List[np.ndarray] = []
    for p in planets:
        mda, mdd = relative_offset_mas(p, t, data["t_ref"], plx_mas)
        model_da.append(mda)
        model_dd.append(mdd)
        d2.append(((da - mda) / se_da) ** 2 + ((dd - mdd) / se_dd) ** 2)

    d2_arr = np.vstack(d2)
    d2_min = np.min(d2_arr, axis=0)
    nearest_planet = np.argmin(d2_arr, axis=0) + 1
    likely = d2_min < IMG_LIKELY_D2_THRESH

    rows = []
    epoch_summary = []
    for ie in range(data["N_img_epoch"]):
        m = int(data["Mi"][ie])
        k0 = int(data["Eidx"][ie])
        if m <= 0:
            epoch_summary.append(dict(epoch=ie, Mi=m, assignment="no detections"))
            continue
        if m == 1:
            idx = k0
            ap = int(nearest_planet[idx])
            j = ap - 1
            rows.append(dict(
                epoch=ie,
                local_detection="A",
                row_index=idx,
                t_jd=t[idx],
                da_obs_mas=da[idx],
                ddec_obs_mas=dd[idx],
                sigma_da_mas=sda[idx],
                sigma_ddec_mas=sdd[idx],
                sigma_eff_da_mas=se_da[idx],
                sigma_eff_ddec_mas=se_dd[idx],
                assigned_planet=ap,
                assignment_rule="nearest-single-detection",
                model_da_mas=model_da[j][idx],
                model_ddec_mas=model_dd[j][idx],
                residual_da_mas=da[idx] - model_da[j][idx],
                residual_ddec_mas=dd[idx] - model_dd[j][idx],
                d2_assigned=d2_arr[j, idx],
                d2_planet1=d2_arr[0, idx] if len(planets) >= 1 else np.nan,
                d2_planet2=d2_arr[1, idx] if len(planets) >= 2 else np.nan,
                likely_planet_source=bool(likely[idx]),
                possible_background=False,
            ))
            epoch_summary.append(dict(epoch=ie, Mi=m, assignment=f"A->P{ap}", d2=float(d2_arr[j, idx])))
        elif m == 2 and len(planets) >= 2:
            iA, iB = k0, k0 + 1
            L12 = d2_arr[0, iA] + d2_arr[1, iB]
            L21 = d2_arr[1, iA] + d2_arr[0, iB]
            if L12 <= L21:
                assignment = [(iA, "A", 1), (iB, "B", 2)]
                tag = "A->P1,B->P2"
                total_d2 = L12
            else:
                assignment = [(iA, "A", 2), (iB, "B", 1)]
                tag = "A->P2,B->P1"
                total_d2 = L21
            epoch_summary.append(dict(epoch=ie, Mi=m, assignment=tag, d2=float(total_d2)))
            for idx, local, ap in assignment:
                j = ap - 1
                rows.append(dict(
                    epoch=ie,
                    local_detection=local,
                    row_index=idx,
                    t_jd=t[idx],
                    da_obs_mas=da[idx],
                    ddec_obs_mas=dd[idx],
                    sigma_da_mas=sda[idx],
                    sigma_ddec_mas=sdd[idx],
                    sigma_eff_da_mas=se_da[idx],
                    sigma_eff_ddec_mas=se_dd[idx],
                    assigned_planet=ap,
                    assignment_rule="best-two-detection-permutation",
                    model_da_mas=model_da[j][idx],
                    model_ddec_mas=model_dd[j][idx],
                    residual_da_mas=da[idx] - model_da[j][idx],
                    residual_ddec_mas=dd[idx] - model_dd[j][idx],
                    d2_assigned=d2_arr[j, idx],
                    d2_planet1=d2_arr[0, idx],
                    d2_planet2=d2_arr[1, idx],
                    likely_planet_source=bool(likely[idx]),
                    possible_background=False,
                ))
        else:
            # General fallback for unexpected Mi or nplanet: assign each row to nearest planet.
            for local_i in range(m):
                idx = k0 + local_i
                ap = int(nearest_planet[idx])
                j = ap - 1
                rows.append(dict(
                    epoch=ie,
                    local_detection=chr(ord("A") + local_i),
                    row_index=idx,
                    t_jd=t[idx],
                    da_obs_mas=da[idx],
                    ddec_obs_mas=dd[idx],
                    sigma_da_mas=sda[idx],
                    sigma_ddec_mas=sdd[idx],
                    sigma_eff_da_mas=se_da[idx],
                    sigma_eff_ddec_mas=se_dd[idx],
                    assigned_planet=ap,
                    assignment_rule="nearest-fallback",
                    model_da_mas=model_da[j][idx],
                    model_ddec_mas=model_dd[j][idx],
                    residual_da_mas=da[idx] - model_da[j][idx],
                    residual_ddec_mas=dd[idx] - model_dd[j][idx],
                    d2_assigned=d2_arr[j, idx],
                    d2_planet1=d2_arr[0, idx] if len(planets) >= 1 else np.nan,
                    d2_planet2=d2_arr[1, idx] if len(planets) >= 2 else np.nan,
                    likely_planet_source=bool(likely[idx]),
                    possible_background=False,
                ))
            epoch_summary.append(dict(epoch=ie, Mi=m, assignment="nearest-fallback"))

    assignments = pd.DataFrame(rows).sort_values("row_index").reset_index(drop=True) if rows else pd.DataFrame()
    return dict(
        assignments=assignments,
        epoch_summary=pd.DataFrame(epoch_summary),
        t=t,
        da=da,
        dd=dd,
        sda=sda,
        sdd=sdd,
        se_da=se_da,
        se_dd=se_dd,
        model_da=model_da,
        model_dd=model_dd,
        d2_arr=d2_arr,
        d2_min=d2_min,
        nearest_planet=nearest_planet,
        likely=likely,
    )


# ============================================================
# 6. Tables and statistics
# ============================================================
def sampled_parameter_names(nplanet: int) -> List[str]:
    names: List[str] = []
    for k in range(nplanet):
        p = k + 1
        names.extend([
            f"P{p}_day",
            f"e{p}",
            f"Mp{p}_Mjup",
            f"tau{p}_frac",
            f"omega{p}_planet_deg",
            f"cosi{p}",
            f"Omega{p}_deg",
        ])
    names.extend(["parallax_mas", "sigma_jit_AL_mas", "sigma_jit_IMG_mas"])
    return names


def fit_statistics(al_model: pd.DataFrame,
                   n_sampled_pars: int,
                   n_profile_pars: int = 4) -> Dict[str, float]:
    used = al_model["used_in_likelihood_like_C"].to_numpy(dtype=bool)
    r = al_model.loc[used, "al_residual_mas"].to_numpy(dtype=float)
    sig = al_model.loc[used, "sigma_eff_mas"].to_numpy(dtype=float)
    n = len(r)
    chi2 = float(np.sum((r / sig) ** 2)) if n else np.nan
    dof = max(1, n - n_sampled_pars - n_profile_pars)
    rms = float(np.sqrt(np.mean(r * r))) if n else np.nan
    wrms = float(np.sqrt(np.sum((r * r) / (sig * sig)) / np.sum(1.0 / (sig * sig)))) if n else np.nan
    return dict(
        n_AL_used=float(n),
        n_AL_total=float(len(al_model)),
        n_sampled_parameters=float(n_sampled_pars),
        n_profile_parameters=float(n_profile_pars),
        chi2_AL=chi2,
        reduced_chi2_AL=chi2 / dof if n else np.nan,
        rms_AL_residual_mas=rms,
        wrms_AL_residual_mas=wrms,
        median_sigma_eff_AL_mas=float(np.median(sig)) if n else np.nan,
        max_abs_AL_residual_mas=float(np.max(np.abs(r))) if n else np.nan,
    )


def save_tables(outdir: Path,
                params: np.ndarray,
                chain_res: Dict,
                planets: List[Dict],
                plx_mas: float,
                sigma_jit_AL: float,
                sigma_jit_IMG: float,
                al_model: pd.DataFrame,
                x4: np.ndarray,
                ls_info: Dict[str, float],
                stats: Dict[str, float],
                Mstar_Msun: float,
                img_diag: Dict,
                data: Dict) -> None:
    names = sampled_parameter_names(len(planets))
    npars = len(names)
    table = pd.DataFrame({
        "parameter": names,
        "best": params[:npars],
        "median_after_burnin": chain_res["p50"][:npars],
        "p16_after_burnin": chain_res["p16"][:npars],
        "p84_after_burnin": chain_res["p84"][:npars],
    })
    table.to_csv(outdir / "bestfit_sampled_parameters.csv", index=False)

    prof = pd.DataFrame({
        "parameter": ["dalpha0_star_mas", "ddelta0_mas", "pmra_star_masyr", "pmdec_masyr"],
        "profile_best": x4,
    })
    for key, val in ls_info.items():
        prof[key] = val
    prof.to_csv(outdir / "bestfit_profile4p_astrometry.csv", index=False)

    derived = []
    for p in planets:
        alpha_star_mas = p["f_star"] * p["a_rel_AU"] * plx_mas
        K = calc_K_mps(Mstar_Msun, p["Mp_Mjup"], p["P_day"], p["e"], p["cosi"])
        derived.append(dict(
            planet=p["index"],
            P_day=p["P_day"],
            e=p["e"],
            Mp_Mjup=p["Mp_Mjup"],
            tau_frac=p["tau_frac"],
            omega_planet_deg=p["omega_planet_deg"],
            cosi=p["cosi"],
            i_deg_from_abs_cosi=math.degrees(math.acos(min(1.0, max(-1.0, abs(p["cosi"]))))),
            Omega_deg=p["Omega_deg"],
            Mstar_Msun_used=Mstar_Msun,
            parallax_mas=plx_mas,
            a_rel_AU=p["a_rel_AU"],
            a_rel_mas=p["a_rel_AU"] * plx_mas,
            mass_fraction_star_reflex=p["f_star"],
            alpha_star_mas=alpha_star_mas,
            K_mps=K,
        ))
    pd.DataFrame(derived).to_csv(outdir / "bestfit_derived_planet_quantities.csv", index=False)

    al_model.to_csv(outdir / "bestfit_AL_model_timeseries.csv", index=False)

    assign = img_diag.get("assignments", pd.DataFrame())
    if isinstance(assign, pd.DataFrame) and not assign.empty:
        assign.to_csv(outdir / "bestfit_IMG_assignments.csv", index=False)
    epoch_summary = img_diag.get("epoch_summary", pd.DataFrame())
    if isinstance(epoch_summary, pd.DataFrame) and not epoch_summary.empty:
        epoch_summary.to_csv(outdir / "bestfit_IMG_epoch_assignments.csv", index=False)

    stats_all = dict(stats)
    stats_all.update(dict(
        best_chain_row_index=float(chain_res["best_idx"]),
        best_loglike=float(chain_res["best_loglike"]),
        burnin_start_row=float(chain_res["burnin_start"]),
        parallax_mas=float(plx_mas),
        sigma_jit_AL_mas=float(sigma_jit_AL),
        sigma_jit_IMG_mas=float(sigma_jit_IMG),
        IMG_all_detections_confirmed=1.0,
        N_img_total=float(data["N_img_total"]),
        N_img_epoch=float(data["N_img_epoch"]),
    ))
    if isinstance(assign, pd.DataFrame) and not assign.empty:
        stats_all.update(dict(
            rms_IMG_residual_da_mas=float(np.sqrt(np.mean(assign["residual_da_mas"].to_numpy(dtype=float) ** 2))),
            rms_IMG_residual_ddec_mas=float(np.sqrt(np.mean(assign["residual_ddec_mas"].to_numpy(dtype=float) ** 2))),
            median_IMG_d2_assigned=float(np.median(assign["d2_assigned"].to_numpy(dtype=float))),
            count_IMG_assigned_sources=float(len(assign)),
        ))
    pd.DataFrame([stats_all]).to_csv(outdir / "fit_statistics.csv", index=False)

    # Human-readable quick summary.
    lines = []
    def add(s: str = "") -> None:
        lines.append(s)

    add("================ AL + IMG profile-4p plotting summary ================")
    add(f"t_ref_jd = {data['t_ref']:.12f}")
    add(f"N_AL = {data['N_al']}; N_IMG_total = {data['N_img_total']}; N_IMG_epoch = {data['N_img_epoch']}")
    add("IMG source treatment = all detections are confirmed companion-relative astrometry")
    add(f"best chain row index = {chain_res['best_idx']}")
    add(f"best loglike = {chain_res['best_loglike']:.12g}")
    add(f"Mstar_Msun = {Mstar_Msun:.12g}")
    add(f"parallax_mas = {plx_mas:.12g}")
    add(f"sigma_jit_AL_mas = {sigma_jit_AL:.12g}")
    add(f"sigma_jit_IMG_mas = {sigma_jit_IMG:.12g}")
    add("")
    add("Profiled 4p astrometry:")
    for name, val in zip(["dalpha0_star_mas", "ddelta0_mas", "pmra_star_masyr", "pmdec_masyr"], x4):
        add(f"  {name:20s} = {val:+.12g}")
    add("")
    add("AL fit statistics:")
    for key in ["n_AL_used", "chi2_AL", "reduced_chi2_AL", "rms_AL_residual_mas", "wrms_AL_residual_mas"]:
        add(f"  {key:28s} = {stats_all.get(key, np.nan):.12g}")
    if isinstance(assign, pd.DataFrame) and not assign.empty:
        add("")
        add("IMG assignment diagnostics:")
        for _, r in assign.iterrows():
            add(
                f"  epoch {int(r['epoch']):3d} row {int(r['row_index']):3d} {r['local_detection']} -> P{int(r['assigned_planet'])}: "
                f"t={r['t_jd']:.6f}, obs=({r['da_obs_mas']:.4f},{r['ddec_obs_mas']:.4f}) mas, "
                f"model=({r['model_da_mas']:.4f},{r['model_ddec_mas']:.4f}) mas, "
                f"resid=({r['residual_da_mas']:.4f},{r['residual_ddec_mas']:.4f}) mas, "
                f"d2={r['d2_assigned']:.4g}"
            )
    (outdir / "bestfit_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# 7. AL plotting functions
# ============================================================
def save_al_raw_time_series(model: pd.DataFrame, outdir: Path) -> None:
    t = model["obs_time_tcb"].to_numpy(dtype=float)
    y = model["centroid_pos_al_mas"].to_numpy(dtype=float)
    sig = model["centroid_pos_error_al_mas"].to_numpy(dtype=float)
    used = model["used_in_likelihood_like_C"].to_numpy(dtype=bool)
    order = np.argsort(t)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.errorbar(t[used], y[used], yerr=sig[used], fmt=".", ms=4, alpha=0.75, label="AL data used")
    if np.any(~used):
        ax.scatter(t[~used], y[~used], marker="x", s=28, label="flagged/outlier")
    ax.plot(t[order], model["al_model_mas"].to_numpy()[order], lw=1.4, label="profile4p + planets model")
    ax.plot(t[order], model["al_single_model_mas"].to_numpy()[order], lw=1.0, alpha=0.85, label="profile4p single-star + parallax part")
    ax.set_xlabel("JD (TCB)")
    ax.set_ylabel("centroid_pos_al (mas)")
    ax.set_title("Raw Gaia-like AL time series")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "01_al_raw_time_series.png", dpi=220)
    plt.close(fig)


def save_al_orbital_signal(model: pd.DataFrame, outdir: Path, nplanet: int) -> None:
    t = model["obs_time_tcb"].to_numpy(dtype=float)
    used = model["used_in_likelihood_like_C"].to_numpy(dtype=bool)
    y_orb = model["al_obs_minus_single_mas"].to_numpy(dtype=float)
    order = np.argsort(t)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.errorbar(
        t[used], y_orb[used],
        yerr=model.loc[used, "sigma_eff_mas"].to_numpy(dtype=float),
        fmt=".", ms=4, alpha=0.75,
        label="data - profile4p single-star part",
    )
    ax.plot(t[order], model["al_orbit_model_mas"].to_numpy()[order], lw=1.5, label="combined planet AL model")
    for k in range(nplanet):
        ax.plot(t[order], model[f"al_orbit_model_p{k+1}_mas"].to_numpy()[order], lw=1.0, alpha=0.85, label=f"planet {k+1}")
    ax.axhline(0.0, lw=0.8)
    ax.set_xlabel("JD (TCB)")
    ax.set_ylabel("AL orbital signal (mas)")
    ax.set_title("AL orbital signal after subtracting profile-4p stellar terms")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "02_al_orbital_signal_time_series.png", dpi=220)
    plt.close(fig)


def save_residual_plots(model: pd.DataFrame, outdir: Path) -> None:
    t = model["obs_time_tcb"].to_numpy(dtype=float)
    r = model["al_residual_mas"].to_numpy(dtype=float)
    sig = model["sigma_eff_mas"].to_numpy(dtype=float)
    used = model["used_in_likelihood_like_C"].to_numpy(dtype=bool)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.errorbar(t[used], r[used], yerr=sig[used], fmt=".", ms=4, alpha=0.8)
    ax.axhline(0.0, lw=0.8)
    ax.set_xlabel("JD (TCB)")
    ax.set_ylabel("AL residual (mas)")
    ax.set_title("AL residuals")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "03_al_residuals_time_series.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.hist(r[used], bins=30, alpha=0.8)
    ax.axvline(0.0, lw=0.8)
    ax.set_xlabel("AL residual (mas)")
    ax.set_ylabel("count")
    ax.set_title("AL residual histogram")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "04_al_residual_histogram.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(model.loc[used, "scanAngle_rad"], r[used], s=16, alpha=0.75)
    ax.axhline(0.0, lw=0.8)
    ax.set_xlabel("scanAngle_rad")
    ax.set_ylabel("AL residual (mas)")
    ax.set_title("AL residual vs scan angle")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "07_residual_vs_scanAngle.png", dpi=220)
    plt.close(fig)

    if model["ccd_row"].notna().any():
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.scatter(model.loc[used, "ccd_row"], r[used], s=16, alpha=0.75)
        ax.axhline(0.0, lw=0.8)
        ax.set_xlabel("ccd_row")
        ax.set_ylabel("AL residual (mas)")
        ax.set_title("AL residual vs CCD row")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "08_residual_vs_ccd_row.png", dpi=220)
        plt.close(fig)

    if model["fov_code"].notna().any():
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.scatter(model.loc[used, "fov_code"], r[used], s=16, alpha=0.75)
        ax.axhline(0.0, lw=0.8)
        ax.set_xlabel("fov_code")
        ax.set_ylabel("AL residual (mas)")
        ax.set_title("AL residual vs FoV")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "09_residual_vs_fov.png", dpi=220)
        plt.close(fig)


def dense_reflex_track(planets: List[Dict],
                       t_ref: float,
                       plx_mas: float,
                       t_min: float,
                       t_max: float,
                       ngrid: int = ORBIT_NGRID) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    t_dense = np.linspace(t_min, t_max, ngrid)
    dra_total = np.zeros_like(t_dense)
    ddec_total = np.zeros_like(t_dense)
    per_full = []
    for p in planets:
        # Full ellipse for each planet.
        phase_t = t_ref + np.linspace(0.0, max(p["P_day"], 1e-12), ngrid)
        dra_full, ddec_full = stellar_reflex_offset_mas(p, phase_t, t_ref, plx_mas)
        per_full.append((dra_full, ddec_full))
        dra_d, ddec_d = stellar_reflex_offset_mas(p, t_dense, t_ref, plx_mas)
        dra_total += dra_d
        ddec_total += ddec_d
    return t_dense, dra_total, ddec_total, per_full


def save_al_skyplane_plot(model: pd.DataFrame,
                          planets: List[Dict],
                          t_ref: float,
                          plx_mas: float,
                          outdir: Path,
                          show_constraint_lines: bool = True,
                          line_half_length_mas: float = 1.5) -> None:
    t = model["obs_time_tcb"].to_numpy(dtype=float)
    dra = model["dalpha_orbit_total_mas"].to_numpy(dtype=float)
    ddec = model["ddelta_orbit_total_mas"].to_numpy(dtype=float)
    xp = model["dalpha_pseudopoint_mas"].to_numpy(dtype=float)
    yp = model["ddelta_pseudopoint_mas"].to_numpy(dtype=float)
    psi = model["scanAngle_rad"].to_numpy(dtype=float)
    used = model["used_in_likelihood_like_C"].to_numpy(dtype=bool)

    _, dra_dense, ddec_dense, per_full = dense_reflex_track(planets, t_ref, plx_mas, float(np.min(t)), float(np.max(t)))


    plt.rcParams.update({
        'font.size': 18,          # 全局基础字体大小
        'axes.titlesize': 20,     # 标题字体大小
        'axes.labelsize': 14,     # 坐标轴标签字体大小
        'xtick.labelsize': 13,    # X轴刻度字体大小
        'ytick.labelsize': 13,    # Y轴刻度字体大小
        'legend.fontsize': 16,    # 图例字体大小
    })

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(dra_dense, ddec_dense, lw=1.5, label="combined reflex track over observed span")
    for k, (dra_one, ddec_one) in enumerate(per_full):
        ax.plot(dra_one, ddec_one, lw=1.0, alpha=0.75, label=f"planet {k+1} full reflex ellipse")
    ax.scatter(dra[used], ddec[used], s=20, label="best-fit AL epoch model points", zorder=3)
    ax.scatter(xp[used], yp[used], marker="x", s=24, label="AL pseudo-points", zorder=4)

    if show_constraint_lines:
        half = float(line_half_length_mas)
        n_ra = np.cos(psi)
        n_dec = -np.sin(psi)
        step = max(1, len(t) // 150)
        for i in range(0, len(t), step):
            if not used[i]:
                continue
            x0, y0 = xp[i], yp[i]
            dx, dy = half * n_ra[i], half * n_dec[i]
            ax.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], lw=0.7, alpha=0.25)

    ax.axhline(0.0, lw=0.8)
    ax.axvline(0.0, lw=0.8)
    ax.set_xlabel(r"stellar reflex $\Delta\alpha^\ast$ (mas)")
    ax.set_ylabel(r"stellar reflex $\Delta\delta$ (mas)")
    ax.set_title("Sky-plane stellar reflex orbit constrained by AL scans")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(loc="best",fontsize = 16, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "05_al_skyplane_reflex_orbit.png", dpi=240)
    plt.close(fig)



def save_al_skyplane_and_residuals_combined_plot(model: pd.DataFrame,
                                                 planets: List[Dict],
                                                 t_ref: float,
                                                 plx_mas: float,
                                                 outdir: Path,
                                                 show_constraint_lines: bool = True,
                                                 line_half_length_mas: float = 1.5) -> None:
    """
    Combined diagnostic figure:
      top    : same content as 05_al_skyplane_reflex_orbit.png
      bottom : same content as 03_al_residuals_time_series.png

    The original single-panel figures are still saved by their original functions.
    """
    # ---------- Top panel: sky-plane reflex orbit ----------
    t = model["obs_time_tcb"].to_numpy(dtype=float)
    dra = model["dalpha_orbit_total_mas"].to_numpy(dtype=float)
    ddec = model["ddelta_orbit_total_mas"].to_numpy(dtype=float)
    xp = model["dalpha_pseudopoint_mas"].to_numpy(dtype=float)
    yp = model["ddelta_pseudopoint_mas"].to_numpy(dtype=float)
    psi = model["scanAngle_rad"].to_numpy(dtype=float)
    used = model["used_in_likelihood_like_C"].to_numpy(dtype=bool)

    _, dra_dense, ddec_dense, per_full = dense_reflex_track(
        planets, t_ref, plx_mas, float(np.min(t)), float(np.max(t))
    )

    # ---------- Bottom panel: AL residual time series ----------
    r = model["al_residual_mas"].to_numpy(dtype=float)
    sig = model["sigma_eff_mas"].to_numpy(dtype=float)

    plt.rcParams.update({
        'font.size': 18,          # 全局基础字体大小
        'axes.titlesize': 20,     # 标题字体大小
        'axes.labelsize': 14,     # 坐标轴标签字体大小
        'xtick.labelsize': 13,    # X轴刻度字体大小
        'ytick.labelsize': 13,    # Y轴刻度字体大小
        'legend.fontsize': 12,    # 图例字体大小
    })


    fig, (ax_sky, ax_res) = plt.subplots(
        2, 1,
        figsize=(9.0, 10.0),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.3},
    )

    ax_sky.plot(dra_dense, ddec_dense, lw=1.5, label="combined reflex track over observed span")
    for k, (dra_one, ddec_one) in enumerate(per_full):
        ax_sky.plot(dra_one, ddec_one, lw=1.0, alpha=0.75, label=f"{object_letter_label(k)} full reflex ellipse")
    ax_sky.scatter(dra[used], ddec[used], s=20, label="best-fit AL epoch model points", zorder=3)
    ax_sky.scatter(xp[used], yp[used], marker="x", s=24, label="AL pseudo-points", zorder=4)

    if show_constraint_lines:
        half = float(line_half_length_mas)
        n_ra = np.cos(psi)
        n_dec = -np.sin(psi)
        step = max(1, len(t) // 150)
        for i in range(0, len(t), step):
            if not used[i]:
                continue
            x0, y0 = xp[i], yp[i]
            dx, dy = half * n_ra[i], half * n_dec[i]
            ax_sky.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], lw=0.7, alpha=0.25)

    ax_sky.axhline(0.0, lw=0.8)
    ax_sky.axvline(0.0, lw=0.8)
    ax_sky.set_xlabel(r"stellar reflex $\Delta\alpha^\ast$ (mas)")
    ax_sky.set_ylabel(r"stellar reflex $\Delta\delta$ (mas)")
    # ax_sky.set_title("Sky-plane stellar reflex orbit constrained by AL scans")
    ax_sky.set_aspect("equal", adjustable="box")
    ax_sky.grid(alpha=0.3)
    ax_sky.legend(loc="best")
    ax_sky.set_xlim(-8, 6)

    ax_res.errorbar(t[used], r[used], yerr=sig[used], fmt=".", ms=4, alpha=0.8)
    ax_res.axhline(0.0, lw=0.8)
    ax_res.set_xlabel("JD (TCB)")
    ax_res.set_ylabel("AL residual (mas)")
    ax_res.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "05_al_skyplane_reflex_orbit_with_residuals.png", dpi=240)
    plt.close(fig)

def smooth_periodic_curve(phase: np.ndarray, value: np.ndarray, ngrid: int, window: int) -> Tuple[np.ndarray, np.ndarray]:
    phase = np.asarray(phase, dtype=float)
    value = np.asarray(value, dtype=float)
    ok = np.isfinite(phase) & np.isfinite(value)
    phase = phase[ok]
    value = value[ok]
    grid = np.linspace(0.0, 1.0, ngrid)
    if len(phase) < 3:
        return grid, np.full_like(grid, np.nan)
    order = np.argsort(phase)
    ph = phase[order]
    val = value[order]
    ph_ext = np.concatenate([ph - 1.0, ph, ph + 1.0])
    val_ext = np.concatenate([val, val, val])
    interp = np.interp(grid, ph_ext, val_ext)
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    w = min(w, max(3, len(grid) // 3 * 2 + 1))
    kernel = np.ones(w, dtype=float) / w
    pad = w // 2
    interp_pad = np.concatenate([interp[-pad:], interp, interp[:pad]])
    sm = np.convolve(interp_pad, kernel, mode="valid")
    return grid, sm[:len(grid)]


def save_phase_folded_plots(model: pd.DataFrame,
                            planets: List[Dict],
                            t_ref: float,
                            outdir: Path) -> None:
    t = model["obs_time_tcb"].to_numpy(dtype=float)
    y_total = model["al_obs_minus_single_mas"].to_numpy(dtype=float)
    sig = model["sigma_eff_mas"].to_numpy(dtype=float)
    used = model["used_in_likelihood_like_C"].to_numpy(dtype=bool)

    for k, p in enumerate(planets):
        pnum = k + 1
        phase = np.mod(p["tau_frac"] + (t - t_ref) / p["P_day"], 1.0)
        model_k = model[f"al_orbit_model_p{pnum}_mas"].to_numpy(dtype=float)
        other = np.zeros_like(y_total)
        for j in range(len(planets)):
            if j != k:
                other += model[f"al_orbit_model_p{j+1}_mas"].to_numpy(dtype=float)
        y_k = y_total - other
        grid, smooth_model = smooth_periodic_curve(phase[used], model_k[used], PHASE_NGRID, PHASE_SMOOTH_WINDOW)

        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.errorbar(
            phase[used], y_k[used], yerr=sig[used], fmt=".", ms=4, alpha=0.75,
            label="data - profile4p - other planets",
        )
        ax.scatter(phase[used], model_k[used], s=12, alpha=0.35, label="model at observed scan angles")
        if np.any(np.isfinite(smooth_model)):
            ax.plot(grid, smooth_model, lw=1.8, label="smoothed visual guide")
            ax.plot(grid + 1.0, smooth_model, lw=1.8)
        ax.axhline(0.0, lw=0.8)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("mean-anomaly phase at t_ref")
        ax.set_ylabel("AL signal (mas)")
        ax.set_title(f"Phase-folded AL signal for planet {pnum}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / f"06_phase_folded_planet_{pnum}.png", dpi=220)
        plt.close(fig)


# ============================================================
# 8. IMG plotting functions
# ============================================================
def full_relative_orbit(p: Dict, t_ref: float, plx_mas: float, ngrid: int = ORBIT_NGRID) -> Tuple[np.ndarray, np.ndarray]:
    tt = t_ref + np.linspace(0.0, max(p["P_day"], 1e-12), ngrid)
    return relative_offset_mas(p, tt, t_ref, plx_mas)


def save_img_diagnostics(outdir: Path,
                         data: Dict,
                         planets: List[Dict],
                         plx_mas: float,
                         img_diag: Dict) -> None:
    """
    Clean one-panel direct-imaging figure.

    This replaces the older 2x2 IMG diagnostic figure and the separate IMG
    residual figures.  The purpose is to make a compact paper-style plot that
    shows only the relative companion orbits, IMG measurements, model positions
    at the observed epochs, and a few fit-quality numbers.
    """
    assignments: pd.DataFrame = img_diag.get("assignments", pd.DataFrame())
    t = img_diag.get("t", np.array([]))
    if len(t) == 0:
        return

    da = img_diag["da"]
    dd = img_diag["dd"]
    sda = img_diag["sda"]
    sdd = img_diag["sdd"]

    # Use a small, consistent style for the IMG plot only.
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    })

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])

    fig, ax = plt.subplots(figsize=(8.2, 8.0))

    # ---------- Best-fit relative orbits ----------
    all_x: List[float] = []
    all_y: List[float] = []
    for ip, p in enumerate(planets):
        color = colors[ip % len(colors)]
        dra_orb, ddec_orb = full_relative_orbit(p, data["t_ref"], plx_mas)
        ax.plot(dra_orb, ddec_orb, lw=2.0, color=color, alpha=0.92)
        all_x.extend(np.asarray(dra_orb, dtype=float).tolist())
        all_y.extend(np.asarray(ddec_orb, dtype=float).tolist())

        # # Mark the model position at t_ref; this gives the reader an orientation
        # # without adding a long legend.
        # dra0, ddec0 = relative_offset_mas(p, np.array([data["t_ref"]]), data["t_ref"], plx_mas)
        # ax.scatter(dra0, ddec0, marker="s", s=34, color=color, zorder=4)

        # Inline orbit label with the key orbital quantities.
        lab_idx = int(0.58 * (len(dra_orb) - 1))
        a_rel_mas = p["a_rel_AU"] * plx_mas
        label = (
            f"P{p['index']}: $P$={p['P_day']:.1f} d\n"
            f"$e$={p['e']:.2f}, $M_p$={p['Mp_Mjup']:.2g} $M_J$\n"
            f"$a$={a_rel_mas:.1f} mas"
        )
        ax.annotate(
            label,
            xy=(dra_orb[lab_idx], ddec_orb[lab_idx]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=12,
            color=color,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.72),
        )

    # ---------- IMG measurements and model-at-epoch positions ----------
    # If the assignment table exists, color measurements by assigned planet and
    # draw short O-C segments.  Otherwise fall back to plotting all IMG points.
    if not assignments.empty:
        assignments = assignments.sort_values("row_index").reset_index(drop=True)
        n_annotate = False

        for ip, p in enumerate(planets):
            color = colors[ip % len(colors)]
            aa = assignments[assignments["assigned_planet"].astype(int) == int(p["index"])]
            if aa.empty:
                continue

            bg = aa["possible_background"].to_numpy(dtype=bool) if "possible_background" in aa else np.zeros(len(aa), dtype=bool)
            aa_planet = aa.loc[~bg]
            aa_bg = aa.loc[bg]

            # Main planet-associated detections: solid green points.
            if not aa_planet.empty:
                ax.errorbar(
                    aa_planet["da_obs_mas"], aa_planet["ddec_obs_mas"],
                    xerr=aa_planet["sigma_da_mas"], yerr=aa_planet["sigma_ddec_mas"],
                    fmt="o", ms=4.2, capsize=2.0, lw=0.8,
                    color="limegreen", mfc="limegreen", mec="limegreen", mew=0.9,
                    alpha=0.98, zorder=5,
                )
                ax.scatter(
                    aa_planet["model_da_mas"], aa_planet["model_ddec_mas"],
                    marker="o", s=78, facecolors="none", edgecolors="black", linewidths=1.5, zorder=6,
                )

            # Possible background/noise detections are deliberately muted.
            if not aa_bg.empty:
                ax.errorbar(
                    aa_bg["da_obs_mas"], aa_bg["ddec_obs_mas"],
                    xerr=aa_bg["sigma_da_mas"], yerr=aa_bg["sigma_ddec_mas"],
                    fmt="o", ms=4.0, capsize=2.0, lw=0.8,
                    color="0.55", mfc="white", mec="0.55", mew=1.0,
                    alpha=0.75, zorder=4,
                )

            # Thin observed-to-model segments.  These show the residual direction
            # but avoid a separate residual subplot.
            for _, r in aa.iterrows():
                seg_color = "0.55" if bool(r.get("possible_background", False)) else color
                ax.plot(
                    [r["da_obs_mas"], r["model_da_mas"]],
                    [r["ddec_obs_mas"], r["model_ddec_mas"]],
                    ls="--", lw=0.7, color=seg_color, alpha=0.45, zorder=3,
                )
                all_x.extend([float(r["da_obs_mas"]), float(r["model_da_mas"])])
                all_y.extend([float(r["ddec_obs_mas"]), float(r["model_ddec_mas"])])

                # Keep labels compact: epoch number only, and only for small-N IMG sets.
                if n_annotate:
                    ax.text(
                        float(r["da_obs_mas"]), float(r["ddec_obs_mas"]),
                        f" {int(r['epoch'])}", fontsize=8.5, color=color,
                        ha="left", va="bottom", alpha=0.85, zorder=7,
                    )
    else:
        ax.errorbar(
            da, dd, xerr=sda, yerr=sdd,
            fmt="o", ms=4.2, capsize=2.0, lw=0.8,
            color="limegreen", mfc="limegreen", mec="limegreen", mew=0.9,
            alpha=0.98, zorder=5,
        )
        all_x.extend(np.asarray(da, dtype=float).tolist())
        all_y.extend(np.asarray(dd, dtype=float).tolist())

    # ---------- Star and compact explanatory marks ----------
    ax.scatter([0.0], [0.0], marker="*", s=150, color="black", zorder=8)
    ax.text(0.0, 0.0, "  host star", fontsize=12, ha="left", va="center", color="black")

    # Clean legend with styled colored handles.
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor="limegreen", markeredgecolor="limegreen",
               label="IMG data"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=9,
               markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.5,
               label="Model at IMG epochs"),
        Line2D([0], [0], marker="*", linestyle="None", markersize=11,
               markerfacecolor="black", markeredgecolor="black",
               label="Host star"),
    ]
    for ip, p in enumerate(planets):
        color = colors[ip % len(colors)]
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2.2, label=object_orbit_label(ip))
        )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fancybox=True,
        framealpha=0.92,
        borderpad=0.6,
        labelspacing=0.5,
        handlelength=1.8,
        handletextpad=0.7,
        fontsize=13,
    )

    # Fit-quality text box.  These are useful numbers but do not require extra panels.
    summary_lines = []
    if not assignments.empty:
        rda = assignments["residual_da_mas"].to_numpy(dtype=float)
        rdd = assignments["residual_ddec_mas"].to_numpy(dtype=float)
        rms_2d = np.sqrt(np.mean(rda * rda + rdd * rdd))
        rms_da = np.sqrt(np.mean(rda * rda))
        rms_dd = np.sqrt(np.mean(rdd * rdd))
        summary_lines.extend([
            rf"RMS$_{{2D}}$ = {rms_2d:.3g} mas",
            rf"RMS$_{{\alpha^\ast,\delta}}$ = {rms_da:.3g}, {rms_dd:.3g} mas",
        ])
    ax.text(
        0.97, 0.03, "\n".join(summary_lines),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.70", alpha=0.82),
    )

    # ---------- Axes and output ----------
    ax.axhline(0.0, lw=0.8, color="0.65", zorder=1)
    ax.axvline(0.0, lw=0.8, color="0.65", zorder=1)
    ax.set_xlabel(r"relative $\Delta\alpha^\ast$ (mas, East +)")
    ax.set_ylabel(r"relative $\Delta\delta$ (mas, North +)")
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    # Apply a robust margin using all orbit, data, and model points.
    xx = np.asarray([v for v in all_x if np.isfinite(v)], dtype=float)
    yy = np.asarray([v for v in all_y if np.isfinite(v)], dtype=float)
    if len(xx) > 0 and len(yy) > 0:
        xmin, xmax = float(np.min(xx)), float(np.max(xx))
        ymin, ymax = float(np.min(yy)), float(np.max(yy))
        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)
        pad = 0.10 * max(dx, dy)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)


    fig.tight_layout()
    fig.savefig(outdir / "10_img_relative_orbits_clean.png", dpi=260)
    plt.close(fig)


def remove_old_png_outputs(outdir: Path) -> None:
    """Remove old PNG figures so each run leaves only the requested combined figure."""
    for old_png in outdir.glob("*.png"):
        try:
            old_png.unlink()
        except OSError:
            pass


def save_al_img_wide_combined_figure(model: pd.DataFrame,
                                     planets: List[Dict],
                                     data: Dict,
                                     plx_mas: float,
                                     img_diag: Dict,
                                     outdir: Path,
                                     show_constraint_lines: bool = True,
                                     line_half_length_mas: float = 1.5,
                                     filename: str = COMBINED_FIGURE_NAME) -> None:
    """
    Paper-style wide figure with three panels:
      upper-left  : AL sky-plane stellar-reflex orbit, without AL O-C subplot
      upper-right : clean IMG relative-orbit plot
      bottom      : AL O-C residuals spanning both upper panels

    No panel titles are used. Margins are intentionally small for LaTeX use.
    The upper-panel outer frames are kept aligned with the lower O-C frame.
    """
    # ---------- Shared style ----------
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    })
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])

    fig = plt.figure(figsize=(17.2, 9.0))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[3.35, 1.05],
        width_ratios=[1.0, 1.0],
        hspace=0.20,
        wspace=0.18,
    )
    ax_al = fig.add_subplot(gs[0, 0])
    ax_img = fig.add_subplot(gs[0, 1])
    ax_res = fig.add_subplot(gs[1, :])

    # ========================================================
    # Upper-left: AL sky-plane stellar-reflex orbit
    # ========================================================
    t = model["obs_time_tcb"].to_numpy(dtype=float)
    dra = model["dalpha_orbit_total_mas"].to_numpy(dtype=float)
    ddec = model["ddelta_orbit_total_mas"].to_numpy(dtype=float)
    xp = model["dalpha_pseudopoint_mas"].to_numpy(dtype=float)
    yp = model["ddelta_pseudopoint_mas"].to_numpy(dtype=float)
    psi = model["scanAngle_rad"].to_numpy(dtype=float)
    used = model["used_in_likelihood_like_C"].to_numpy(dtype=bool)

    _, dra_dense, ddec_dense, per_full = dense_reflex_track(
        planets, data["t_ref"], plx_mas, float(np.nanmin(t)), float(np.nanmax(t))
    )

    ax_al.plot(dra_dense, ddec_dense, lw=2.7, label="combined reflex track")
    for k, (dra_one, ddec_one) in enumerate(per_full):
        ax_al.plot(dra_one, ddec_one, lw=1.9, alpha=0.82, label=f"{object_letter_label(k)} reflex ellipse")
    ax_al.scatter(dra[used], ddec[used], s=38, linewidths=1.0, label="AL model epochs", zorder=3)
    ax_al.scatter(xp[used], yp[used], marker="x", s=52, linewidths=1.6, label="AL pseudo-points", zorder=4)

    if show_constraint_lines:
        half = float(line_half_length_mas)
        n_ra = np.cos(psi)
        n_dec = -np.sin(psi)
        step = max(1, len(t) // 150)
        for i in range(0, len(t), step):
            if not used[i]:
                continue
            x0, y0 = xp[i], yp[i]
            dx, dy = half * n_ra[i], half * n_dec[i]
            ax_al.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], lw=1.05, alpha=0.30)

    ax_al.axhline(0.0, lw=1.2, color="0.60", zorder=1)
    ax_al.axvline(0.0, lw=1.2, color="0.60", zorder=1)
    ax_al.set_xlabel(r"stellar reflex $\Delta\alpha^\ast$ (mas)")
    ax_al.set_ylabel(r"stellar reflex $\Delta\delta$ (mas)")
    # Keep the top-panel frame aligned with the bottom O-C panel.
    # adjustable="datalim" preserves the GridSpec axes box and expands
    # the data limits instead of shrinking the axes frame.
    ax_al.set_aspect("equal", adjustable="datalim")
    ax_al.grid(alpha=0.30, linewidth=0.75)
    ax_al.legend(loc="best",frameon=False, fancybox=True, framealpha=0.90,
                 borderpad=0.45, labelspacing=0.35, handlelength=1.45)

    # Robust AL panel limits with small padding.
    al_x = np.asarray([v for v in np.concatenate([dra_dense, dra[used], xp[used]]) if np.isfinite(v)], dtype=float)
    al_y = np.asarray([v for v in np.concatenate([ddec_dense, ddec[used], yp[used]]) if np.isfinite(v)], dtype=float)
    if len(al_x) > 0 and len(al_y) > 0:
        xmin, xmax = float(np.min(al_x)), float(np.max(al_x))
        ymin, ymax = float(np.min(al_y)), float(np.max(al_y))
        pad = 0.055 * max(xmax - xmin, ymax - ymin, 1e-6)
        ax_al.set_xlim(xmin - pad, xmax + pad)
        ax_al.set_ylim(ymin - pad, ymax + pad)

    # ========================================================
    # Upper-right: IMG relative orbit figure, clean version
    # ========================================================
    assignments: pd.DataFrame = img_diag.get("assignments", pd.DataFrame())
    t_img = img_diag.get("t", np.array([]))
    all_x: List[float] = []
    all_y: List[float] = []

    for ip, p in enumerate(planets):
        color = colors[ip % len(colors)]
        dra_orb, ddec_orb = full_relative_orbit(p, data["t_ref"], plx_mas)
        ax_img.plot(dra_orb, ddec_orb, lw=2.9, color=color, alpha=0.94)
        all_x.extend(np.asarray(dra_orb, dtype=float).tolist())
        all_y.extend(np.asarray(ddec_orb, dtype=float).tolist())

        lab_idx = int(0.58 * (len(dra_orb) - 1))
        a_rel_mas = p["a_rel_AU"] * plx_mas
        label = (
            f"P{p['index']}: $P$={p['P_day']:.1f} d\n"
            f"$e$={p['e']:.2f}, $M_p$={p['Mp_Mjup']:.2g} $M_J$\n"
            f"$a$={a_rel_mas:.1f} mas"
        )
        ax_img.annotate(
            label,
            xy=(dra_orb[lab_idx], ddec_orb[lab_idx]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10.5,
            color=color,
            bbox=dict(boxstyle="round,pad=0.23", fc="white", ec=color, alpha=0.70),
        )

    if len(t_img) > 0:
        if not assignments.empty:
            assignments = assignments.sort_values("row_index").reset_index(drop=True)
            for ip, p in enumerate(planets):
                color = colors[ip % len(colors)]
                aa = assignments[assignments["assigned_planet"].astype(int) == int(p["index"])]
                if aa.empty:
                    continue
                bg = aa["possible_background"].to_numpy(dtype=bool) if "possible_background" in aa else np.zeros(len(aa), dtype=bool)
                aa_planet = aa.loc[~bg]
                aa_bg = aa.loc[bg]

                if not aa_planet.empty:
                    ax_img.errorbar(
                        aa_planet["da_obs_mas"], aa_planet["ddec_obs_mas"],
                        xerr=aa_planet["sigma_da_mas"], yerr=aa_planet["sigma_ddec_mas"],
                        fmt="o", ms=6.0, capsize=3.0, lw=1.2, elinewidth=1.2, capthick=1.2,
                        color="limegreen", mfc="limegreen", mec="limegreen", mew=1.15,
                        alpha=0.98, zorder=5,
                    )
                    ax_img.scatter(
                        aa_planet["model_da_mas"], aa_planet["model_ddec_mas"],
                        marker="o", s=125, facecolors="none", edgecolors="black",
                        linewidths=2.25, zorder=6,
                    )

                if not aa_bg.empty:
                    ax_img.errorbar(
                        aa_bg["da_obs_mas"], aa_bg["ddec_obs_mas"],
                        xerr=aa_bg["sigma_da_mas"], yerr=aa_bg["sigma_ddec_mas"],
                        fmt="o", ms=5.5, capsize=3.0, lw=1.1, elinewidth=1.1, capthick=1.1,
                        color="0.55", mfc="white", mec="0.55", mew=1.2,
                        alpha=0.75, zorder=4,
                    )

                for _, rr in aa.iterrows():
                    seg_color = "0.55" if bool(rr.get("possible_background", False)) else color
                    ax_img.plot(
                        [rr["da_obs_mas"], rr["model_da_mas"]],
                        [rr["ddec_obs_mas"], rr["model_ddec_mas"]],
                        ls="--", lw=1.1, color=seg_color, alpha=0.52, zorder=3,
                    )
                    all_x.extend([float(rr["da_obs_mas"]), float(rr["model_da_mas"])])
                    all_y.extend([float(rr["ddec_obs_mas"]), float(rr["model_ddec_mas"])])
        else:
            da = img_diag["da"]
            dd = img_diag["dd"]
            sda = img_diag["sda"]
            sdd = img_diag["sdd"]
            ax_img.errorbar(
                da, dd, xerr=sda, yerr=sdd,
                fmt="o", ms=6.0, capsize=3.0, lw=1.2, elinewidth=1.2, capthick=1.2,
                color="limegreen", mfc="limegreen", mec="limegreen", mew=1.15,
                alpha=0.98, zorder=5,
            )
            all_x.extend(np.asarray(da, dtype=float).tolist())
            all_y.extend(np.asarray(dd, dtype=float).tolist())

    ax_img.scatter([0.0], [0.0], marker="*", s=210, color="black", zorder=8)
    ax_img.text(0.0, 0.0, "  host star", fontsize=11, ha="left", va="center", color="black")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=8.2,
               markerfacecolor="limegreen", markeredgecolor="limegreen", label="IMG data"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=12.0,
               markerfacecolor="none", markeredgecolor="black", markeredgewidth=2.1,
               label="Model at IMG epochs"),
        Line2D([0], [0], marker="*", linestyle="None", markersize=12.0,
               markerfacecolor="black", markeredgecolor="black", label="Host star")
    ]
    for ip, p in enumerate(planets):
        color = colors[ip % len(colors)]
        legend_handles.append(Line2D([0], [0], color=color, lw=2.9, label=object_orbit_label(ip)))
    ax_img.legend(handles=legend_handles, loc="upper left",frameon=False, fancybox=True,
                  framealpha=0.92, borderpad=0.45, labelspacing=0.35,
                  handlelength=1.55, handletextpad=0.55)

    summary_lines = []
    if not assignments.empty:
        rda = assignments["residual_da_mas"].to_numpy(dtype=float)
        rdd = assignments["residual_ddec_mas"].to_numpy(dtype=float)
        rms_2d = np.sqrt(np.mean(rda * rda + rdd * rdd))
        rms_da = np.sqrt(np.mean(rda * rda))
        rms_dd = np.sqrt(np.mean(rdd * rdd))
        summary_lines.extend([
            rf"RMS$_{{2D}}$ = {rms_2d:.3g} mas",
            rf"RMS$_{{\alpha^\ast,\delta}}$ = {rms_da:.3g}, {rms_dd:.3g} mas",
        ])
    if summary_lines:
        ax_img.text(
            0.97, 0.03, "\n".join(summary_lines),
            transform=ax_img.transAxes, ha="right", va="bottom", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.30", fc="white", ec="0.70", alpha=0.82),
        )

    ax_img.axhline(0.0, lw=1.2, color="0.60", zorder=1)
    ax_img.axvline(0.0, lw=1.2, color="0.60", zorder=1)
    ax_img.set_xlabel(r"relative $\Delta\alpha^\ast$ (mas, East +)")
    ax_img.set_ylabel(r"relative $\Delta\delta$ (mas, North +)")
    ax_img.grid(alpha=0.30, linewidth=0.75)
    # Same reason as the AL panel: preserve the axes box so the upper
    # right frame aligns with the full-width bottom O-C frame.
    ax_img.set_aspect("equal", adjustable="datalim")

    xx = np.asarray([v for v in all_x if np.isfinite(v)], dtype=float)
    yy = np.asarray([v for v in all_y if np.isfinite(v)], dtype=float)
    if len(xx) > 0 and len(yy) > 0:
        xmin, xmax = float(np.min(xx)), float(np.max(xx))
        ymin, ymax = float(np.min(yy)), float(np.max(yy))
        pad = 0.065 * max(xmax - xmin, ymax - ymin, 1e-6)
        ax_img.set_xlim(xmin - pad, xmax + pad)
        ax_img.set_ylim(ymin - pad, ymax + pad)

    # ========================================================
    # Bottom: AL O-C residuals spanning the full width
    # ========================================================
    r = model["al_residual_mas"].to_numpy(dtype=float)
    sig = model["sigma_eff_mas"].to_numpy(dtype=float)
    ax_res.errorbar(t[used], r[used], yerr=sig[used], fmt="o", ms=4.2, lw=1.15, elinewidth=1.15, capsize=0,
                    alpha=0.86)
    ax_res.axhline(0.0, lw=1.35, color="0.30")
    ax_res.set_xlabel("JD (TCB)")
    ax_res.set_ylabel("AL O-C (mas)")
    ax_res.grid(alpha=0.30, linewidth=0.75)
    if np.any(used):
        xmin, xmax = float(np.nanmin(t[used])), float(np.nanmax(t[used]))
        pad = 0.015 * max(xmax - xmin, 1e-6)
        ax_res.set_xlim(xmin - pad, xmax + pad)

    # Thicker axes and tick marks for print readability.
    for ax in (ax_al, ax_img, ax_res):
        for spine in ax.spines.values():
            spine.set_linewidth(1.25)
        ax.tick_params(axis="both", which="major", width=1.25, length=5.5)
        ax.tick_params(axis="both", which="minor", width=1.0, length=3.5)

    # Make the figure occupy almost the whole canvas; useful for paper insertion.
    fig.subplots_adjust(left=0.045, right=0.992, top=0.992, bottom=0.075,
                        wspace=0.105, hspace=0.205)
    fig.savefig(outdir / filename, dpi=320, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


# ============================================================
# 9. CLI and main
# ============================================================
def make_args_from_internal_config() -> argparse.Namespace:
    return argparse.Namespace(
        data_matrix=DATA_MATRIX,
        chain=CHAIN_FILE,
        input_ini=INPUT_INI,
        outdir=OUTDIR,
        nplanet=NPLANET,
        loglike_col=LOGLIKE_COL,
        param_start_col=PARAM_START_COL,
        burnin_fraction=BURNIN_FRACTION,
        show_constraint_lines=SHOW_AL_CONSTRAINT_LINES,
        line_half_length_mas=LINE_HALF_LENGTH_MAS,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AL+IMG profile-4p joint-fit diagnostics.")
    parser.add_argument("--use_cli", action="store_true", help="Use command-line arguments instead of the script-top configuration.")
    parser.add_argument("--data_matrix", default=DATA_MATRIX)
    parser.add_argument("--chain", default=CHAIN_FILE)
    parser.add_argument("--input_ini", default=INPUT_INI)
    parser.add_argument("--outdir", default=OUTDIR)
    parser.add_argument("--nplanet", type=int, default=NPLANET)
    parser.add_argument("--loglike_col", type=int, default=LOGLIKE_COL)
    parser.add_argument("--param_start_col", type=int, default=PARAM_START_COL)
    parser.add_argument("--burnin_fraction", type=float, default=BURNIN_FRACTION)
    parser.add_argument("--show_constraint_lines", action="store_true")
    parser.add_argument("--no_constraint_lines", action="store_true")
    parser.add_argument("--line_half_length_mas", type=float, default=LINE_HALF_LENGTH_MAS)

    if USE_INTERNAL_CONFIG_BY_DEFAULT and len(sys.argv) == 1:
        return make_args_from_internal_config()
    args = parser.parse_args()
    if USE_INTERNAL_CONFIG_BY_DEFAULT and not args.use_cli:
        return make_args_from_internal_config()
    if args.no_constraint_lines:
        args.show_constraint_lines = False
    elif not args.show_constraint_lines:
        args.show_constraint_lines = bool(SHOW_AL_CONSTRAINT_LINES)
    return args


def main() -> None:
    args = parse_args()
    data_matrix = Path(args.data_matrix)
    chain_path = Path(args.chain)
    input_ini = Path(args.input_ini)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not data_matrix.exists():
        raise FileNotFoundError(f"Cannot find DATA_MATRIX: {data_matrix.resolve()}")
    if not chain_path.exists():
        raise FileNotFoundError(f"Cannot find CHAIN_FILE: {chain_path.resolve()}")

    nplanet = int(args.nplanet)
    if nplanet != 2:
        raise ValueError("This plotting script is written for the current two-planet C likelihood. Set NPLANET=2.")

    Mstar_in_g = get_mstar_in_g(input_ini)
    Mstar_Msun = Mstar_in_g / MSUN_G

    mat = read_matrix(data_matrix)
    data = parse_matrix(mat)
    npars = 7 * nplanet + 3
    chain_res = read_chain(
        chain_path,
        npars=npars,
        param_start_col=int(args.param_start_col),
        loglike_col=int(args.loglike_col),
        burnin_fraction=float(args.burnin_fraction),
    )
    params = chain_res["best_row"][int(args.param_start_col):int(args.param_start_col) + npars]

    planets = unpack_planets(params, Mstar_Msun, nplanet)
    plx_mas = float(params[7 * nplanet + 0])
    sigma_jit_AL = float(params[7 * nplanet + 1])
    sigma_jit_IMG = float(params[7 * nplanet + 2])

    al_df = make_al_dataframe(data["al"])
    al_model, x4, ls_info = evaluate_al_model(al_df, planets, data["t_ref"], plx_mas, sigma_jit_AL)
    img_diag = compute_img_diagnostics(data, planets, plx_mas, sigma_jit_IMG)
    stats = fit_statistics(al_model, n_sampled_pars=npars, n_profile_pars=4)

    save_tables(
        outdir=outdir,
        params=params,
        chain_res=chain_res,
        planets=planets,
        plx_mas=plx_mas,
        sigma_jit_AL=sigma_jit_AL,
        sigma_jit_IMG=sigma_jit_IMG,
        al_model=al_model,
        x4=x4,
        ls_info=ls_info,
        stats=stats,
        Mstar_Msun=Mstar_Msun,
        img_diag=img_diag,
        data=data,
    )

    # Remove old PNG outputs from previous runs and save only the requested wide figure.
    remove_old_png_outputs(outdir)
    save_al_img_wide_combined_figure(
        al_model, planets, data, plx_mas, img_diag, outdir,
        show_constraint_lines=bool(args.show_constraint_lines),
        line_half_length_mas=float(args.line_half_length_mas),
    )

    print("=" * 78)
    print("AL + IMG profile-4p plotting finished")
    print(f"data_matrix        = {data_matrix}")
    print(f"chain              = {chain_path}")
    print(f"input_ini          = {input_ini}")
    print(f"outdir             = {outdir}")
    print(f"nplanet            = {nplanet}")
    print(f"t_ref_jd           = {data['t_ref']:.12f}")
    print(f"best row index     = {chain_res['best_idx']}")
    print(f"best loglike       = {chain_res['best_loglike']:.8g}")
    print(f"parallax_mas       = {plx_mas:.12g}")
    print("profile 4p:")
    for name, val in zip(["dalpha0*", "ddelta0", "pmra*", "pmdec"], x4):
        print(f"  {name:10s} = {val:+.12g}")
    print("saved main figure:")
    print(f"  {outdir / COMBINED_FIGURE_NAME}")
    print("=" * 78)


if __name__ == "__main__":
    main()

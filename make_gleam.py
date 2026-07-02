#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from dataclasses import dataclass


MODEL_NAMES = ["AL", "AST", "RV", "AL_IMG", "AL_RV", "AST_IMG", "AST_RV", "IMG_RV"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    per_planet: int
    amp_name: str
    globals_after_planets: list[str]
    needs_mstar: bool = False
    needs_rv_sys: bool = False
    has_al: bool = False
    has_ast: bool = False
    has_img: bool = False
    has_rv: bool = False
    al_profile_dim: int = 0
    has_img_jitter: bool = False
    sampled_plx: bool = False


SPECS = {
    "AL": ModelSpec("AL", 7, "alpha_ast_mas", ["sigma_jit_AL_mas"], has_al=True, al_profile_dim=5),
    "AST": ModelSpec("AST", 7, "Mp_Mjup", ["dra0_star_mas", "ddec0_mas", "pmra_star_masyr", "pmdec_masyr", "plx_mas"], needs_mstar=True, needs_rv_sys=True, has_ast=True, sampled_plx=True),
    "RV": ModelSpec("RV", 5, "K_mps", [], has_rv=True),
    "AL_IMG": ModelSpec("AL_IMG", 7, "Mp_Mjup", ["plx_mas", "sigma_jit_AL_mas", "sigma_jit_IMG_mas"], needs_mstar=True, has_al=True, has_img=True, al_profile_dim=4, has_img_jitter=True, sampled_plx=True),
    "AL_RV": ModelSpec("AL_RV", 7, "K_mps", ["plx_mas", "sigma_jit_AL_mas"], has_al=True, has_rv=True, al_profile_dim=4, sampled_plx=True),
    "AST_IMG": ModelSpec("AST_IMG", 7, "Mp_Mjup", ["dra0_star_mas", "ddec0_mas", "pmra_star_masyr", "pmdec_masyr", "plx_mas"], needs_mstar=True, needs_rv_sys=True, has_ast=True, has_img=True, sampled_plx=True),
    "AST_RV": ModelSpec("AST_RV", 7, "K_mps", ["dra0_star_mas", "ddec0_mas", "pmra_star_masyr", "pmdec_masyr", "plx_mas"], needs_rv_sys=True, has_ast=True, has_rv=True, sampled_plx=True),
    "IMG_RV": ModelSpec("IMG_RV", 7, "K_mps", ["plx_mas"], needs_mstar=True, has_img=True, has_rv=True, sampled_plx=True),
}


def n_parm_for(spec: ModelSpec, n_companions: int, n_rv_sources: int) -> int:
    n = spec.per_planet * n_companions + len(spec.globals_after_planets)
    if spec.has_rv:
        n += n_rv_sources + 1
    return n


def planet_param_names(spec: ModelSpec, k: int) -> list[str]:
    if spec.per_planet == 5:
        return [
            f"P{k}_days",
            f"e{k}",
            f"K{k}_mps",
            f"tau{k}_frac",
            f"omega{k}_planet_deg",
        ]
    return [
        f"P{k}_days",
        f"e{k}",
        f"{spec.amp_name}{k}",
        f"tau{k}_frac",
        f"omega{k}_planet_deg",
        f"cosi{k}",
        f"Omega{k}_deg",
    ]


def parameter_list(spec: ModelSpec, n_companions: int, n_rv_sources: int) -> list[str]:
    names: list[str] = []
    for k in range(1, n_companions + 1):
        names.extend(planet_param_names(spec, k))
    names.extend(spec.globals_after_planets)
    if spec.has_rv:
        for j in range(n_rv_sources):
            names.append(f"rv_offset_source{j}_mps")
        names.append("rv_jitter_mps")
    return names


def common_c_header(n_companions: int, spec: ModelSpec) -> str:
    externs = ["extern int ndim_data;", "extern int N_parm;"]
    if spec.needs_mstar:
        externs.append("extern double Mstar_Msun;")
    if spec.needs_rv_sys:
        externs.append("extern double rv_sys_kms;")
    externs.extend([
        "extern double calc_M_tau(double t_jd, double t_ref_jd, double period_days, double tau_frac);",
        "extern double newton_solver(double M, double e);",
        "extern double calc_nv(double EE, double ecc);",
        "extern double calc_RV(double K, double nv, double omega, double ecc);",
        "extern double period_mass_to_au(double period_days, double mtot_Msun);",
        "extern double alpha_star_mas_from_K(double K_mps, double period_days, double ecc, double cosi, double plx_mas);",
        "extern double solve_mp_mjup_from_K(double K_mps, double period_days, double ecc, double cosi, double ms_Msun);",
    ])
    if spec.has_al:
        externs.append("extern int solve_linear_system(int n, double *A, double *b, double *x);")

    img_helpers = 'static double img_assignment_rec(int d, int m, int nplanet, const double *logp, int *used)\n{{\n    int k;\n    double best = -INFINITY;\n    double vals[MAX_PLANET];\n    int nv = 0;\n\n    if (d == m) return 0.0;\n\n    for (k = 0; k < nplanet; ++k) {{\n        if (!used[k]) {{\n            double v;\n            used[k] = 1;\n            v = logp[d * nplanet + k] + img_assignment_rec(d + 1, m, nplanet, logp, used);\n            used[k] = 0;\n            vals[nv++] = v;\n            if (v > best) best = v;\n        }}\n    }}\n\n    if (!isfinite(best)) return -INFINITY;\n    {{\n        double s = 0.0;\n        for (k = 0; k < nv; ++k) s += exp(vals[k] - best);\n        return best + log(s);\n    }}\n}}\n\nstatic double img_epoch_logsum(const double *logp, int m, int nplanet)\n{{\n    int used[MAX_PLANET];\n    int k;\n    if (m < 0 || m > nplanet || nplanet > MAX_PLANET) return -INFINITY;\n    if (m == 0) return 0.0;\n    for (k = 0; k < MAX_PLANET; ++k) used[k] = 0;\n    return img_assignment_rec(0, m, nplanet, logp, used);\n}}\n' if spec.has_img else ""
    ast_helpers = 'static inline void ast_normalize_vec3(double p[3])\n{{\n    double r = sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);\n    if (!(r > 0.0) || !isfinite(r)) r = 1.0;\n    p[0] /= r;\n    p[1] /= r;\n    p[2] /= r;\n}}\n\nstatic inline void ast_apply_first_order_aberration(double p[3], double vx_kms, double vy_kms, double vz_kms)\n{{\n    double beta[3] = { vx_kms / 299792.458, vy_kms / 299792.458, vz_kms / 299792.458 };\n    double pdotb = p[0]*beta[0] + p[1]*beta[1] + p[2]*beta[2];\n    double q[3];\n    q[0] = p[0] + beta[0] - pdotb * p[0];\n    q[1] = p[1] + beta[1] - pdotb * p[1];\n    q[2] = p[2] + beta[2] - pdotb * p[2];\n    p[0] = q[0];\n    p[1] = q[1];\n    p[2] = q[2];\n    ast_normalize_vec3(p);\n}}\n\n' if spec.name in ("AST", "AST_IMG") else ""
    sky_helpers = 'static void thiele_innes(double rho_mas, double omega_deg, double Omega_deg, double cosi,\n                         double *A, double *B, double *F, double *G)\n{{\n    double omega = omega_deg * DEG2RAD;\n    double Omega = Omega_deg * DEG2RAD;\n    double cO = cos(Omega), sO = sin(Omega);\n    double cw = cos(omega), sw = sin(omega);\n\n    *A = rho_mas * ( cw * cO - sw * sO * cosi);\n    *B = rho_mas * ( cw * sO + sw * cO * cosi);\n    *F = rho_mas * (-sw * cO - cw * sO * cosi);\n    *G = rho_mas * (-sw * sO + cw * cO * cosi);\n}}\n\nstatic int sky_offset_from_orbit(double t_jd, double t_ref_jd, const OrbitPack *p,\n                                 double scale_mas, int stellar_reflex,\n                                 double *dalpha_star_mas, double *ddelta_mas)\n{{\n    double A, B, F, G, M, E, X, Y, rho;\n\n    M = calc_M_tau(t_jd, t_ref_jd, p->P_day, p->tau_frac);\n    E = newton_solver(M, p->e);\n    if (!isfinite(E)) return 0;\n\n    X = cos(E) - p->e;\n    Y = sqrt(fmax(0.0, 1.0 - p->e * p->e)) * sin(E);\n    rho = stellar_reflex ? -scale_mas : scale_mas;\n    thiele_innes(rho, p->omega_planet_deg, p->Omega_deg, p->cosi, &A, &B, &F, &G);\n\n    *dalpha_star_mas = B * X + G * Y;\n    *ddelta_mas      = A * X + F * Y;\n    return isfinite(*dalpha_star_mas) && isfinite(*ddelta_mas);\n}}\n\n' if (spec.has_al or spec.has_ast or spec.has_img) else ""
    rv_helpers = 'static double rv_for_planet(double t_jd, double t_ref_jd, const OrbitPack *p)\n{{\n    double M, E, nu, omega_star;\n    M = calc_M_tau(t_jd, t_ref_jd, p->P_day, p->tau_frac);\n    E = newton_solver(M, p->e);\n    if (!isfinite(E)) return NAN;\n    nu = calc_nv(E, p->e);\n    omega_star = (p->omega_planet_deg + 180.0) * DEG2RAD;\n    return calc_RV(p->amp, nu, omega_star, p->e);\n}}\n\n' if spec.has_rv else ""

    return f'''#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "alloc.h"
#include "sofa.h"
#include "sofam.h"

{chr(10).join(externs)}

#ifndef NPLANET
#define NPLANET {n_companions}
#endif

#ifndef MAX_PLANET
#define MAX_PLANET 16
#endif

#ifndef MAX_INST
#define MAX_INST 32
#endif

#ifndef PMRA_IS_MU_ALPHA_STAR
#define PMRA_IS_MU_ALPHA_STAR 1
#endif

#ifndef USE_OUTLIER_FLAG
#define USE_OUTLIER_FLAG 1
#endif

#ifndef PI
#define PI 3.141592653589793238462643383279502884
#endif

#define TWOPI   (2.0 * PI)
#define DEG2RAD (PI / 180.0)
#define MAS2RAD (1.0e-3 * DAS2R)
#define MJUP_TO_MSUN 0.000954588
#define DAT(row,col) data_NlineNdim[(row) * ndim_data + (col)]

typedef struct OrbitPack {{
    double P_day;
    double e;
    double amp;
    double tau_frac;
    double omega_planet_deg;
    double cosi;
    double Omega_deg;
    double alpha_rel_mas;
    double alpha_star_mas;
}} OrbitPack;

static inline double sqr(double x) {{ return x * x; }}

static double log_normal_1d(double res, double sig)
{{
    if (!(sig > 0.0) || !isfinite(sig)) return -INFINITY;
    return -0.5 * log(TWOPI * sig * sig) - 0.5 * sqr(res / sig);
}}

static int basic_planet_ok(const OrbitPack *p, int need_sky)
{{
    if (!(p->P_day > 0.0) || !isfinite(p->P_day)) return 0;
    if (!(p->e >= 0.0 && p->e < 1.0) || !isfinite(p->e)) return 0;
    if (!(p->amp >= 0.0) || !isfinite(p->amp)) return 0;
    if (need_sky) {{
        if (!(p->cosi >= -1.0 && p->cosi <= 1.0) || !isfinite(p->cosi)) return 0;
    }}
    return 1;
}}

{ast_helpers}
{sky_helpers}
{rv_helpers}
{img_helpers}\n'''


def c_parse_planets(spec: ModelSpec) -> str:
    if spec.per_planet == 5:
        return '''    for (ip = 0; ip < NPLANET; ++ip) {
        int q = 5 * ip;
        pl[ip].P_day = ptr_one_chain[q + 0];
        pl[ip].e = ptr_one_chain[q + 1];
        pl[ip].amp = ptr_one_chain[q + 2];
        pl[ip].tau_frac = ptr_one_chain[q + 3];
        pl[ip].omega_planet_deg = ptr_one_chain[q + 4];
        pl[ip].cosi = 0.0;
        pl[ip].Omega_deg = 0.0;
        pl[ip].alpha_rel_mas = 0.0;
        pl[ip].alpha_star_mas = 0.0;
        if (!basic_planet_ok(&pl[ip], 0)) return -INFINITY;
    }
'''
    return '''    for (ip = 0; ip < NPLANET; ++ip) {
        int q = 7 * ip;
        pl[ip].P_day = ptr_one_chain[q + 0];
        pl[ip].e = ptr_one_chain[q + 1];
        pl[ip].amp = ptr_one_chain[q + 2];
        pl[ip].tau_frac = ptr_one_chain[q + 3];
        pl[ip].omega_planet_deg = ptr_one_chain[q + 4];
        pl[ip].cosi = ptr_one_chain[q + 5];
        pl[ip].Omega_deg = ptr_one_chain[q + 6];
        pl[ip].alpha_rel_mas = 0.0;
        pl[ip].alpha_star_mas = 0.0;
        if (!basic_planet_ok(&pl[ip], 1)) return -INFINITY;
    }
'''


def c_scale_setup(spec: ModelSpec) -> str:
    lines = []
    if spec.needs_mstar:
        lines.append("    if (!(Mstar_Msun > 0.0) || !isfinite(Mstar_Msun)) return -INFINITY;\n")
    if spec.sampled_plx:
        lines.append("    if (!(plx_mas > 0.0) || !isfinite(plx_mas)) return -INFINITY;\n")

    if spec.amp_name == "alpha_ast_mas":
        lines.append('''    for (ip = 0; ip < NPLANET; ++ip) {
        pl[ip].alpha_star_mas = pl[ip].amp;
    }
''')
    elif spec.amp_name == "Mp_Mjup":
        lines.append('''    for (ip = 0; ip < NPLANET; ++ip) {
        double mp_sun = pl[ip].amp * MJUP_TO_MSUN;
        double a_rel_AU = period_mass_to_au(pl[ip].P_day, Mstar_Msun + mp_sun);
        pl[ip].alpha_rel_mas = a_rel_AU * plx_mas;
        pl[ip].alpha_star_mas = pl[ip].alpha_rel_mas * mp_sun / (Mstar_Msun + mp_sun);
        if (!isfinite(pl[ip].alpha_rel_mas) || !isfinite(pl[ip].alpha_star_mas)) return -INFINITY;
    }
''')
    elif spec.amp_name == "K_mps" and (spec.has_al or spec.has_ast or spec.has_img):
        lines.append('''    for (ip = 0; ip < NPLANET; ++ip) {
        pl[ip].alpha_star_mas = alpha_star_mas_from_K(pl[ip].amp, pl[ip].P_day,
                                                      pl[ip].e, pl[ip].cosi, plx_mas);
        if (!isfinite(pl[ip].alpha_star_mas)) return -INFINITY;
''')
        if spec.has_img:
            lines.append('''        {
            double mp_mjup = solve_mp_mjup_from_K(pl[ip].amp, pl[ip].P_day,
                                                  pl[ip].e, pl[ip].cosi, Mstar_Msun);
            double mp_sun = mp_mjup * MJUP_TO_MSUN;
            double a_rel_AU;
            if (!isfinite(mp_mjup)) return -INFINITY;
            a_rel_AU = period_mass_to_au(pl[ip].P_day, Mstar_Msun + mp_sun);
            pl[ip].alpha_rel_mas = a_rel_AU * plx_mas;
            if (!isfinite(pl[ip].alpha_rel_mas)) return -INFINITY;
        }
''')
        lines.append("    }\n")
    return "".join(lines)


def c_al_section(profile_dim: int, row_prefix: str) -> str:
    if row_prefix == "AL":
        parse = '''    iStart_AL = (int)floor(DAT(0, 0) + 0.5);
    N_al = (int)floor(DAT(0, 1) + 0.5);
    t_ref_jd = DAT(0, 2) + DAT(0, 3);
'''
    elif row_prefix == "AL_IMG":
        parse = '''    iStart_AL = (int)floor(DAT(0, 0) + 0.5);
    iStart_IMG = (int)floor(DAT(0, 1) + 0.5);
    N_img_epoch = (int)floor(DAT(0, 2) + 0.5);
    t_ref_jd = DAT(0, 3) + DAT(0, 4);
    N_al = iStart_IMG - iStart_AL;
'''
    else:
        parse = '''    iStart_AL = (int)floor(DAT(0, 0) + 0.5);
    iStart_RV = (int)floor(DAT(0, 1) + 0.5);
    N_inst = (int)floor(DAT(0, 2) + 0.5);
    t_ref_jd = DAT(0, 3) + DAT(0, 4);
    N_al = iStart_RV - iStart_AL;
'''

    if profile_dim == 5:
        hset = '''        H[0] = sin_psi;
        H[1] = cos_psi;
        H[2] = dt_yr * sin_psi;
        H[3] = dt_yr * cos_psi;
        H[4] = p_al;
        yprime = al_obs_mas - al_orbit_mas;
'''
        model = '''        al_5p_mas = x_lin[0] * sin_psi
                  + x_lin[1] * cos_psi
                  + x_lin[2] * dt_yr * sin_psi
                  + x_lin[3] * dt_yr * cos_psi
                  + x_lin[4] * p_al;
        al_model_mas = al_5p_mas + al_orbit_mas;
'''
    else:
        hset = '''        H[0] = sin_psi;
        H[1] = cos_psi;
        H[2] = dt_yr * sin_psi;
        H[3] = dt_yr * cos_psi;
        yprime = al_obs_mas - al_orbit_mas - plx_mas * p_al;
'''
        model = '''        al_5p_mas = x_lin[0] * sin_psi
                  + x_lin[1] * cos_psi
                  + x_lin[2] * dt_yr * sin_psi
                  + x_lin[3] * dt_yr * cos_psi
                  + plx_mas * p_al;
        al_model_mas = al_5p_mas + al_orbit_mas;
'''

    return f'''    /* AL rows and reference epoch. */
{parse}    if (iStart_AL < 1 || N_al < {profile_dim} || iStart_AL + N_al > nline_data) return -INFINITY;

    for (a = 0; a < {profile_dim}; ++a) {{
        rhs[a] = 0.0;
        x_lin[a] = 0.0;
        for (b = 0; b < {profile_dim}; ++b) normal[a * {profile_dim} + b] = 0.0;
    }}

    for (i = 0; i < N_al; ++i) {{
        int row = iStart_AL + i;
        double t_jd       = DAT(row, 0);
        double al_obs_mas = DAT(row, 1);
        double sigma_mas  = DAT(row, 2);
        double sin_psi    = DAT(row, 4);
        double cos_psi    = DAT(row, 5);
        double p_al       = DAT(row, 6);
        double out_flag   = DAT(row, 7);
        double dt_yr, H[5], al_orbit_mas, yprime, sig_eff, w;
        double da_sum = 0.0, dd_sum = 0.0;

#if USE_OUTLIER_FLAG
        if (out_flag >= 0.5) continue;
#endif
        if (!(sigma_mas > 0.0) || !isfinite(sigma_mas)) return -INFINITY;

        for (ip = 0; ip < NPLANET; ++ip) {{
            double da, dd;
            if (!sky_offset_from_orbit(t_jd, t_ref_jd, &pl[ip], pl[ip].alpha_star_mas, 1, &da, &dd)) return -INFINITY;
            da_sum += da;
            dd_sum += dd;
        }}
        al_orbit_mas = da_sum * sin_psi + dd_sum * cos_psi;
        dt_yr = (t_jd - t_ref_jd) / 365.25;
{hset}        sig_eff = sqrt(sigma_mas * sigma_mas + sigma_jit_AL * sigma_jit_AL);
        if (!(sig_eff > 0.0) || !isfinite(sig_eff)) return -INFINITY;
        w = 1.0 / (sig_eff * sig_eff);

        for (a = 0; a < {profile_dim}; ++a) {{
            rhs[a] += H[a] * w * yprime;
            for (b = 0; b < {profile_dim}; ++b) normal[a * {profile_dim} + b] += H[a] * w * H[b];
        }}
        n_used++;
    }}

    if (n_used < {profile_dim}) return -INFINITY;
    if (!solve_linear_system({profile_dim}, normal, rhs, x_lin)) return -INFINITY;

    for (i = 0; i < N_al; ++i) {{
        int row = iStart_AL + i;
        double t_jd       = DAT(row, 0);
        double al_obs_mas = DAT(row, 1);
        double sigma_mas  = DAT(row, 2);
        double sin_psi    = DAT(row, 4);
        double cos_psi    = DAT(row, 5);
        double p_al       = DAT(row, 6);
        double out_flag   = DAT(row, 7);
        double dt_yr, al_orbit_mas, al_5p_mas, al_model_mas, sig_eff, lp;
        double da_sum = 0.0, dd_sum = 0.0;

#if USE_OUTLIER_FLAG
        if (out_flag >= 0.5) continue;
#endif
        for (ip = 0; ip < NPLANET; ++ip) {{
            double da, dd;
            if (!sky_offset_from_orbit(t_jd, t_ref_jd, &pl[ip], pl[ip].alpha_star_mas, 1, &da, &dd)) return -INFINITY;
            da_sum += da;
            dd_sum += dd;
        }}
        al_orbit_mas = da_sum * sin_psi + dd_sum * cos_psi;
        dt_yr = (t_jd - t_ref_jd) / 365.25;
{model}        sig_eff = sqrt(sigma_mas * sigma_mas + sigma_jit_AL * sigma_jit_AL);
        lp = log_normal_1d(al_obs_mas - al_model_mas, sig_eff);
        if (!isfinite(lp)) return -INFINITY;
        logll_AL += lp;
    }}
'''


def c_ast_section(row_prefix: str) -> str:
    if row_prefix == "AST":
        parse = '''    iStart_AST = (int)floor(DAT(0, 0) + 0.5);
    {
        int iStop_AST = (int)floor(DAT(0, 1) + 0.5);
        N_ast = iStop_AST - iStart_AST;
    }
    t_ref_jd = DAT(0, 2) + DAT(0, 3);
    ep1a = DAT(0, 2);
    ep1b = DAT(0, 3);
    ra00 = DAT(0, 4);
    dec00 = DAT(0, 5);
    use_aberration = (int)floor(DAT(0, 6) + 0.5);
'''
        ra_setup = '''    {
        double cdec00 = cos(dec00);
        if (fabs(cdec00) < 1.0e-12) return -INFINITY;
        ra0 = ra00 + (dra0_mas / cdec00) * MAS2RAD;
        dec0 = dec00 + ddec0_mas * MAS2RAD;
    }
    pm_ra = pmra_masyr * MAS2RAD;
    pm_dec = pmdec_masyr * MAS2RAD;
#if PMRA_IS_MU_ALPHA_STAR
    if (fabs(cos(dec0)) < 1.0e-12) return -INFINITY;
    pm_ra /= cos(dec0);
#endif
'''
        cdec_pm_block = '''        {
            double cdec_pm = cos(dec_pm);
            if (fabs(cdec_pm) < 1.0e-12) cdec_pm = (cdec_pm >= 0.0) ? 1.0e-12 : -1.0e-12;
            da_coord_mas = da_orb_mas / cdec_pm;
        }
'''
        aberr_block = '''        if (use_aberration) {
            double vnorm = sqrt(vob[0]*vob[0] + vob[1]*vob[1] + vob[2]*vob[2]);
            if (vnorm > 0.0 && isfinite(vnorm)) {
                ast_apply_first_order_aberration(p_all, vob[0], vob[1], vob[2]);
            }
        }
        iauC2s(p_all, &ra_final, &dec_final);
'''
    elif row_prefix == "AST_IMG":
        parse = '''    iStart_AST = (int)floor(DAT(0, 0) + 0.5);
    iStart_IMG = (int)floor(DAT(0, 1) + 0.5);
    N_img_epoch = (int)floor(DAT(0, 2) + 0.5);
    t_ref_jd = DAT(0, 3) + DAT(0, 4);
    ep1a = DAT(0, 3);
    ep1b = DAT(0, 4);
    ra00 = DAT(0, 5);
    dec00 = DAT(0, 6);
    use_aberration = (int)floor(DAT(0, 7) + 0.5);
    N_ast = iStart_IMG - iStart_AST;
'''
        ra_setup = '''    {
        double cdec00 = cos(dec00);
        if (fabs(cdec00) < 1.0e-12) return -INFINITY;
        ra0 = ra00 + (dra0_mas / cdec00) * MAS2RAD;
        dec0 = dec00 + ddec0_mas * MAS2RAD;
    }
    pm_ra = pmra_masyr * MAS2RAD;
    pm_dec = pmdec_masyr * MAS2RAD;
#if PMRA_IS_MU_ALPHA_STAR
    if (fabs(cos(dec0)) < 1.0e-12) return -INFINITY;
    pm_ra /= cos(dec0);
#endif
'''
        cdec_pm_block = '''        {
            double cdec_pm = cos(dec_pm);
            if (fabs(cdec_pm) < 1.0e-12) cdec_pm = (cdec_pm >= 0.0) ? 1.0e-12 : -1.0e-12;
            da_coord_mas = da_orb_mas / cdec_pm;
        }
'''
        aberr_block = '''        if (use_aberration) {
            double vnorm = sqrt(vob[0]*vob[0] + vob[1]*vob[1] + vob[2]*vob[2]);
            if (vnorm > 0.0 && isfinite(vnorm)) {
                ast_apply_first_order_aberration(p_all, vob[0], vob[1], vob[2]);
            }
        }
        iauC2s(p_all, &ra_final, &dec_final);
'''
    else:
        parse = '''    iStart_AST = (int)floor(DAT(0, 0) + 0.5);
    iStart_RV = (int)floor(DAT(0, 1) + 0.5);
    N_inst = (int)floor(DAT(0, 2) + 0.5);
    t_ref_jd = DAT(0, 3) + DAT(0, 4);
    ep1a = DAT(0, 3);
    ep1b = DAT(0, 4);
    ra00 = DAT(0, 5);
    dec00 = DAT(0, 6);
    use_aberration = (int)floor(DAT(0, 7) + 0.5);
    N_ast = iStart_RV - iStart_AST;
'''
        ra_setup = '''    ra0 = ra00 + dra0_mas * MAS2RAD;
    dec0 = dec00 + ddec0_mas * MAS2RAD;
    pm_ra = pmra_masyr * MAS2RAD;
    pm_dec = pmdec_masyr * MAS2RAD;
#if PMRA_IS_MU_ALPHA_STAR
    if (fabs(cos(dec0)) < 1.0e-12) return -INFINITY;
    pm_ra /= cos(dec0);
#endif
'''
        cdec_pm_block = '''        if (fabs(cos(dec_pm)) < 1.0e-12) return -INFINITY;
        da_coord_mas = da_orb_mas / cos(dec_pm);
'''
        aberr_block = '''        if (use_aberration) {
            double d2sun, v2, bm1, ppr[3];
            vob[0] /= (CMPS / 1000.0);
            vob[1] /= (CMPS / 1000.0);
            vob[2] /= (CMPS / 1000.0);
            d2sun = sqrt(pob[0]*pob[0] + pob[1]*pob[1] + pob[2]*pob[2]);
            v2 = vob[0]*vob[0] + vob[1]*vob[1] + vob[2]*vob[2];
            bm1 = sqrt(fmax(0.0, 1.0 - v2));
            iauAb(p_all, vob, d2sun, bm1, ppr);
            iauC2s(ppr, &ra_final, &dec_final);
        } else {
            iauC2s(p_all, &ra_final, &dec_final);
        }
'''
    return f'''    /* AST rows and reference epoch. */
{parse}    if (iStart_AST < 1 || N_ast < 1 || iStart_AST + N_ast > nline_data) return -INFINITY;

{ra_setup}    plx_arcsec = plx_mas / 1000.0;
    plx_rad = plx_arcsec * DAS2R;

    for (i = 0; i < N_ast; ++i) {{
        int row = iStart_AST + i;
        double tJD = DAT(row, 0);
        double obs_da = DAT(row, 1);
        double obs_dd = DAT(row, 2);
        double sig_a = DAT(row, 3) * MAS2RAD;
        double sig_d = DAT(row, 4) * MAS2RAD;
        double pob[3], vob[3], p0[3], p_all[3];
        double ep2a, ep2b, ra_pm, dec_pm, pmr2, pmd2, px2, rv2;
        double da_orb_mas = 0.0, dd_orb_mas = 0.0;
        double ra_pm_pl, dec_pm_pl, da_coord_mas, rnorm;
        double ra_final, dec_final, model_da, model_dd, lp;

        if (!(sig_a > 0.0) || !(sig_d > 0.0) || !isfinite(sig_a) || !isfinite(sig_d)) return -INFINITY;
        if (!isfinite(tJD)) return -INFINITY;
        pob[0] = DAT(row, 5); pob[1] = DAT(row, 6); pob[2] = DAT(row, 7);
        vob[0] = DAT(row, 8); vob[1] = DAT(row, 9); vob[2] = DAT(row,10);

        ep2a = floor(tJD);
        ep2b = tJD - ep2a;
        iauPmsafe(ra0, dec0, pm_ra, pm_dec, plx_arcsec, rv_sys_kms,
                  ep1a, ep1b, ep2a, ep2b,
                  &ra_pm, &dec_pm, &pmr2, &pmd2, &px2, &rv2);
        if (!isfinite(ra_pm) || !isfinite(dec_pm)) return -INFINITY;

        for (ip = 0; ip < NPLANET; ++ip) {{
            double da, dd;
            if (!sky_offset_from_orbit(tJD, t_ref_jd, &pl[ip], pl[ip].alpha_star_mas, 1, &da, &dd)) return -INFINITY;
            da_orb_mas += da;
            dd_orb_mas += dd;
        }}

{cdec_pm_block}        ra_pm_pl = ra_pm + da_coord_mas * MAS2RAD;
        dec_pm_pl = dec_pm + dd_orb_mas * MAS2RAD;
        iauS2c(ra_pm_pl, dec_pm_pl, p0);

        p_all[0] = p0[0] - plx_rad * pob[0];
        p_all[1] = p0[1] - plx_rad * pob[1];
        p_all[2] = p0[2] - plx_rad * pob[2];
        rnorm = sqrt(p_all[0]*p_all[0] + p_all[1]*p_all[1] + p_all[2]*p_all[2]);
        if (!(rnorm > 0.0) || !isfinite(rnorm)) return -INFINITY;
        p_all[0] /= rnorm; p_all[1] /= rnorm; p_all[2] /= rnorm;

{aberr_block}
        model_da = iauAnpm(ra_final - ra00) * cos(dec_final);
        model_dd = dec_final - dec00;
        lp = log_normal_1d(obs_da - model_da, sig_a) + log_normal_1d(obs_dd - model_dd, sig_d);
        if (!isfinite(lp)) return -INFINITY;
        logll_AST += lp;
    }}
'''



def c_rv_section(row_prefix: str, n_rv_sources: int) -> str:
    if row_prefix == "RV":
        parse = '''    iStart_RV = (int)floor(DAT(0, 0) + 0.5);
    N_rv = (int)floor(DAT(0, 1) + 0.5);
    N_inst = (int)floor(DAT(0, 2) + 0.5);
    t_ref_jd = DAT(0, 3) + DAT(0, 4);
'''
        nrvset = ""
    elif row_prefix == "AL_RV":
        parse = ""
        nrvset = "    N_rv = nline_data - iStart_RV;\n"
    elif row_prefix == "AST_RV":
        parse = ""
        nrvset = "    N_rv = nline_data - iStart_RV;\n"
    else:
        parse = '''    iStart_RV = (int)floor(DAT(0, 0) + 0.5);
    iStart_IMG = (int)floor(DAT(0, 1) + 0.5);
    N_inst = (int)floor(DAT(0, 2) + 0.5);
    t_ref_jd = DAT(0, 3) + DAT(0, 4);
    N_img_epoch = (int)floor(DAT(0, 5) + 0.5);
'''
        nrvset = "    N_rv = iStart_IMG - iStart_RV;\n"
    return f'''    /* RV rows. */
{parse}{nrvset}    if (N_inst < 1 || N_inst > MAX_INST) return -INFINITY;
    if (N_inst != {n_rv_sources}) return -INFINITY;
    if (iStart_RV < 1 || N_rv < 1 || iStart_RV + N_rv > nline_data) return -INFINITY;

    for (i = 0; i < N_rv; ++i) {{
        int row = iStart_RV + i;
        double tJD = DAT(row, 0);
        double rv_obs = DAT(row, 1);
        double rv_err = DAT(row, 2);
        int inst = (int)floor(DAT(row, 3) + 0.5);
        double rv_model = 0.0;
        double sig2, sig, lp;

        if (inst < 0 || inst >= N_inst) return -INFINITY;
        if (!(rv_err > 0.0)) return -INFINITY;

        for (ip = 0; ip < NPLANET; ++ip) {{
            double v = rv_for_planet(tJD, t_ref_jd, &pl[ip]);
            if (!isfinite(v)) return -INFINITY;
            rv_model += v;
        }}
        rv_model += rv_offset[inst];
        sig2 = rv_err * rv_err + rv_jitter * rv_jitter;
        sig = sqrt(sig2);
        lp = log_normal_1d(rv_obs - rv_model, sig);
        if (!isfinite(lp)) return -INFINITY;
        logll_RV += lp;
    }}
'''


def c_img_section(row_prefix: str, has_img_jitter: bool) -> str:
    if row_prefix == "AL_IMG":
        mi_start = 5
        parse = ""
    elif row_prefix == "AST_IMG":
        mi_start = 8
        parse = ""
    else:
        mi_start = 6
        parse = ""
    jit = "sigma_jit_IMG" if has_img_jitter else "0.0"
    return f'''    /* IMG rows. Confirmed companion relative astrometry. */
{parse}    if (N_img_epoch < 0) return -INFINITY;
    img_row = iStart_IMG;
    for (ie = 0; ie < N_img_epoch; ++ie) {{
        int m = (int)floor(DAT(0, {mi_start} + ie) + 0.5);
        double logp[MAX_PLANET * MAX_PLANET];
        if (m < 0 || m > NPLANET || NPLANET > MAX_PLANET) return -INFINITY;
        if (img_row + m > nline_data) return -INFINITY;

        for (i = 0; i < m; ++i) {{
            int row = img_row + i;
            double tJD = DAT(row, 0);
            double da_obs = DAT(row, 1);
            double dd_obs = DAT(row, 2);
            double sig_a0 = DAT(row, 3);
            double sig_d0 = DAT(row, 4);
            double sig_a = sqrt(sig_a0 * sig_a0 + ({jit}) * ({jit}));
            double sig_d = sqrt(sig_d0 * sig_d0 + ({jit}) * ({jit}));
            int jp;
            if (!(sig_a > 0.0) || !(sig_d > 0.0)) return -INFINITY;

            for (jp = 0; jp < NPLANET; ++jp) {{
                double da_model, dd_model;
                double lp;
                if (!sky_offset_from_orbit(tJD, t_ref_jd, &pl[jp], pl[jp].alpha_rel_mas, 0, &da_model, &dd_model)) return -INFINITY;
                lp = log_normal_1d(da_obs - da_model, sig_a) + log_normal_1d(dd_obs - dd_model, sig_d);
                logp[i * NPLANET + jp] = lp;
            }}
        }}
        logll_IMG += img_epoch_logsum(logp, m, NPLANET);
        if (!isfinite(logll_IMG)) return -INFINITY;
        img_row += m;
    }}
    if (img_row != nline_data) return -INFINITY;
'''


def render_user_logll(spec: ModelSpec, n_companions: int, n_rv_sources: int) -> str:
    c = [common_c_header(n_companions, spec)]

    int_vars = ["ip", "i"]
    if spec.has_al:
        int_vars.extend(["a", "b"])
    if spec.has_img:
        int_vars.extend(["ie", "img_row"])

    decl = []
    decl.append("    OrbitPack pl[MAX_PLANET];\n")
    decl.append(f"    int {', '.join(int_vars)};\n")
    decl.append(f"    int expected_N_parm = {n_parm_for(spec, n_companions, n_rv_sources)};\n")
    decl.append("    double t_ref_jd = 0.0;\n")
    if spec.has_al:
        decl.append("    double logll_AL = 0.0;\n")
    if spec.has_ast:
        decl.append("    double logll_AST = 0.0;\n")
    if spec.has_img:
        decl.append("    double logll_IMG = 0.0;\n")
    if spec.has_rv:
        decl.append("    double logll_RV = 0.0;\n")

    if spec.has_al:
        decl.append("\n    int iStart_AL = 0, N_al = 0;\n")
    if spec.has_ast:
        decl.append("    int iStart_AST = 0, N_ast = 0;\n")
    if spec.has_img:
        decl.append("    int iStart_IMG = 0, N_img_epoch = 0;\n")
    if spec.has_rv:
        decl.append(f"    int iStart_RV = 0, N_rv = 0, N_inst = {n_rv_sources};\n")

    if spec.sampled_plx:
        decl.append("\n    double plx_mas = 0.0;\n")
    if spec.has_al:
        decl.append("    double sigma_jit_AL = 0.0;\n")
    if spec.has_img and "sigma_jit_IMG_mas" in spec.globals_after_planets:
        decl.append("    double sigma_jit_IMG = 0.0;\n")
    if spec.has_rv:
        decl.append("    double rv_offset[MAX_INST];\n    double rv_jitter = 0.0;\n")

    if spec.has_al:
        decl.append("\n    int n_used = 0;\n")
        decl.append("    double normal[64], rhs[8], x_lin[8];\n")
    if spec.has_ast:
        decl.append("\n    double dra0_mas = 0.0, ddec0_mas = 0.0, pmra_masyr = 0.0, pmdec_masyr = 0.0;\n")
        decl.append("    double ra00 = 0.0, dec00 = 0.0, ep1a = 0.0, ep1b = 0.0;\n")
        decl.append("    double ra0 = 0.0, dec0 = 0.0, pm_ra = 0.0, pm_dec = 0.0, plx_arcsec = 0.0, plx_rad = 0.0;\n")
        decl.append("    int use_aberration = 0;\n")

    if spec.has_rv:
        decl.append("\n    for (i = 0; i < MAX_INST; ++i) rv_offset[i] = 0.0;\n")

    c.append("""
double logll_beta(double *ptr_one_chain,
                  int nline_data,
                  double *data_NlineNdim,
                  double beta_one)
{
""" + "".join(decl) + """
    if (NPLANET > MAX_PLANET) return -INFINITY;
    if (N_parm != expected_N_parm) return -INFINITY;

""")
    c.append(c_parse_planets(spec))

    base = spec.per_planet * n_companions
    idx = base
    parse_globals = []
    for g in spec.globals_after_planets:
        if g == "plx_mas":
            parse_globals.append(f"    plx_mas = ptr_one_chain[{idx}];\n")
        elif g == "sigma_jit_AL_mas":
            parse_globals.append(f"    sigma_jit_AL = ptr_one_chain[{idx}];\n    if (!(sigma_jit_AL >= 0.0) || !isfinite(sigma_jit_AL)) return -INFINITY;\n")
        elif g == "sigma_jit_IMG_mas":
            parse_globals.append(f"    sigma_jit_IMG = ptr_one_chain[{idx}];\n    if (!(sigma_jit_IMG >= 0.0) || !isfinite(sigma_jit_IMG)) return -INFINITY;\n")
        elif g == "dra0_star_mas":
            parse_globals.append(f"    dra0_mas = ptr_one_chain[{idx}];\n")
        elif g == "ddec0_mas":
            parse_globals.append(f"    ddec0_mas = ptr_one_chain[{idx}];\n")
        elif g == "pmra_star_masyr":
            parse_globals.append(f"    pmra_masyr = ptr_one_chain[{idx}];\n")
        elif g == "pmdec_masyr":
            parse_globals.append(f"    pmdec_masyr = ptr_one_chain[{idx}];\n")
        idx += 1
    if spec.has_rv:
        parse_globals.append("    if (N_inst < 1 || N_inst > MAX_INST) return -INFINITY;\n")
        parse_globals.append(f"    for (i = 0; i < {n_rv_sources}; ++i) rv_offset[i] = ptr_one_chain[{idx} + i];\n")
        parse_globals.append(f"    rv_jitter = ptr_one_chain[{idx} + {n_rv_sources}];\n    if (!(rv_jitter >= 0.0) || !isfinite(rv_jitter)) return -INFINITY;\n")
    c.append("".join(parse_globals))
    c.append(c_scale_setup(spec))

    if spec.name == "AL":
        c.append(c_al_section(5, "AL"))
    elif spec.name == "AST":
        c.append(c_ast_section("AST"))
    elif spec.name == "RV":
        c.append(c_rv_section("RV", n_rv_sources))
    elif spec.name == "AL_IMG":
        c.append(c_al_section(4, "AL_IMG"))
        c.append(c_img_section("AL_IMG", True))
    elif spec.name == "AL_RV":
        c.append(c_al_section(4, "AL_RV"))
        c.append(c_rv_section("AL_RV", n_rv_sources))
    elif spec.name == "AST_IMG":
        c.append(c_ast_section("AST_IMG"))
        c.append(c_img_section("AST_IMG", False))
    elif spec.name == "AST_RV":
        c.append(c_ast_section("AST_RV"))
        c.append(c_rv_section("AST_RV", n_rv_sources))
    elif spec.name == "IMG_RV":
        c.append(c_rv_section("IMG_RV", n_rv_sources))
        c.append(c_img_section("IMG_RV", False))

    terms = []
    if spec.has_al:
        terms.append("logll_AL")
    if spec.has_ast:
        terms.append("logll_AST")
    if spec.has_img:
        terms.append("logll_IMG")
    if spec.has_rv:
        terms.append("logll_RV")
    c.append("\n    return (" + " + ".join(terms) + ") * beta_one;\n}\n")
    return "".join(c)


def template_al(n_companions: int, n_rv_sources: int) -> str:
    return render_user_logll(SPECS["AL"], n_companions, n_rv_sources)


def template_ast(n_companions: int, n_rv_sources: int) -> str:
    return render_user_logll(SPECS["AST"], n_companions, n_rv_sources)


def template_rv(n_companions: int, n_rv_sources: int) -> str:
    return render_user_logll(SPECS["RV"], n_companions, n_rv_sources)


def template_al_img(n_companions: int, n_rv_sources: int) -> str:
    return render_user_logll(SPECS["AL_IMG"], n_companions, n_rv_sources)


def template_al_rv(n_companions: int, n_rv_sources: int) -> str:
    return render_user_logll(SPECS["AL_RV"], n_companions, n_rv_sources)


def template_ast_img(n_companions: int, n_rv_sources: int) -> str:
    return render_user_logll(SPECS["AST_IMG"], n_companions, n_rv_sources)


def template_ast_rv_fast(n_companions: int, n_rv_sources: int) -> str:
    return render_user_logll(SPECS["AST_RV"], n_companions, n_rv_sources)


def template_img_rv(n_companions: int, n_rv_sources: int) -> str:
    return render_user_logll(SPECS["IMG_RV"], n_companions, n_rv_sources)


TEMPLATE_FUNCS = {
    "AL": template_al,
    "AST": template_ast,
    "RV": template_rv,
    "AL_IMG": template_al_img,
    "AL_RV": template_al_rv,
    "AST_IMG": template_ast_img,
    "AST_RV": template_ast_rv_fast,
    "IMG_RV": template_img_rv,
}


def write_input_ini(outdir: Path, spec: ModelSpec, n_companions: int, n_rv_sources: int) -> None:
    names = parameter_list(spec, n_companions, n_rv_sources)
    lines = [
        "# Chain grid\n",
        "N_iter:                 1000000\n",
        "N_beta:                 8\n",
        "Beta_Values:            0.005, 0.015, 0.04, 0.10, 0.25, 0.50, 0.78, 1.0\n",
        "\n",
        "# Ladder and proposal control\n",
        "Tune_Ladder:            0\n",
        "N_stopTuneLadder:       300000\n",
        "scale_tune_ladder:      0.2\n",
        "zero_stretch:           0.1\n",
        "n_iter_a_stack:         5000\n",
        "n_iter_a_batch_base:    20\n",
        "n_iter_a_batch_rand:    5\n",
        "N_swap:                 1\n",
        "Swapmode:               1\n",
        "N_stoptune:             10000000\n",
        "N_begintune:            0\n",
        "n_iter_in_tune:         1000\n",
        "ar_ok_lower:            0.10\n",
        "ar_ok_upper:            0.45\n",
        "ar_best:                0.23\n",
        "ar_accept_diff:         0.1\n",
        "sigma_scale_half_ratio: 20\n",
        "sigma_scale_min:        0.0000000005\n",
        "sigma_scale_max:        0.2\n",
        "sigma_jumpin_ratio:     300\n",
        "\n",
        "# Initialisation and output\n",
        "i_save_begin:           0\n",
        "init_rand_seed:         9\n",
        "init_gp_ratio:          0.1\n",
        "Fout_Len:               100\n",
        "FoutPre:                chain\n",
        "FoutSuf:                .dat\n",
        "results_dir:            chains\n",
        "\n",
        "# Data\n",
        "Data_file:              input.dat\n",
        "ndim_data:              TODO_FILL\n",
        "Delimiter:              blank\n",
        "\n",
    ]
    if spec.needs_mstar:
        lines.append("Mstar_Msun:             TODO_FILL\n")
    if spec.needs_rv_sys:
        lines.append("rv_sys_kms:             TODO_FILL\n")
    if spec.needs_mstar or spec.needs_rv_sys:
        lines.append("\n")
    lines.append(f"N_parm:                 {len(names)}\n\n")
    lines.append("# Priors\n")
    for i, name in enumerate(names):
        lines.append(f"para{i}_min:              TODO_FILL    # {name}\n")
        lines.append(f"para{i}_max:              TODO_FILL    # {name}\n\n")
    (outdir / "input.ini").write_text("".join(lines), encoding="utf-8")


def row0_text(spec: ModelSpec) -> str:
    name = spec.name
    if name == "AL":
        return "col0 AL_start_row, col1 N_AL, col2 ep1a, col3 ep1b"
    if name == "AST":
        return "col0 AST_start_row, col1 first_row_after_AST, col2 ep1a, col3 ep1b, col4 ra00_rad, col5 dec00_rad, col6 use_aberration"
    if name == "RV":
        return "col0 RV_start_row, col1 N_RV, col2 N_source, col3 ep1a, col4 ep1b"
    if name == "AL_IMG":
        return "col0 AL_start_row, col1 IMG_start_row, col2 N_img_epoch, col3 ep1a, col4 ep1b, col5... Mi for each IMG epoch"
    if name == "AL_RV":
        return "col0 AL_start_row, col1 RV_start_row, col2 N_source, col3 ep1a, col4 ep1b"
    if name == "AST_IMG":
        return "col0 AST_start_row, col1 IMG_start_row, col2 N_img_epoch, col3 ep1a, col4 ep1b, col5 ra00_rad, col6 dec00_rad, col7 use_aberration, col8... Mi"
    if name == "AST_RV":
        return "col0 AST_start_row, col1 RV_start_row, col2 N_source, col3 ep1a, col4 ep1b, col5 ra00_rad, col6 dec00_rad, col7 use_aberration"
    if name == "IMG_RV":
        return "col0 RV_start_row, col1 IMG_start_row, col2 N_source, col3 ep1a, col4 ep1b, col5 N_img_epoch, col6... Mi"
    return ""


def data_rows_text(spec: ModelSpec) -> str:
    parts = []
    if spec.has_al:
        parts.append("AL rows: col0 tJD, col1 AL_mas, col2 AL_err_mas, col3 scanAngle_rad, col4 sin_psi, col5 cos_psi, col6 parallaxFactorAL, col7 outlier_flag, remaining columns optional.")
    if spec.has_ast:
        parts.append("AST rows: col0 tJD, col1 Delta_alpha*_rad, col2 Delta_delta_rad, col3 sigma_alpha*_mas, col4 sigma_delta_mas, col5-7 observer position in AU, col8-10 observer velocity in km/s.")
    if spec.has_rv:
        parts.append("RV rows: col0 tJD, col1 RV_mps, col2 RV_err_mps, col3 inst_id in 0..N_source-1.")
    if spec.has_img:
        parts.append("IMG rows: col0 tJD, col1 Delta_alpha*_mas, col2 Delta_delta_mas, col3 sigma_alpha*_mas, col4 sigma_delta_mas. The offset is companion relative to host star.")
    return "\n".join(f"- {p}" for p in parts)


def write_readme(outdir: Path, spec: ModelSpec, n_companions: int, n_rv_sources: int) -> None:
    names = parameter_list(spec, n_companions, n_rv_sources)
    lines = [
        f"# GLEAM {spec.name} model\n\n",
        f"Compiled companion count: `{n_companions}`.\n\n",
        "## Row 0\n\n",
        f"{row0_text(spec)}.\n\n",
        "The reference epoch is `t_ref = ep1a + ep1b`.\n\n",
        "## Data rows\n\n",
        data_rows_text(spec),
        "\n\n",
    ]
    if spec.has_img:
        lines.append("For IMG epochs, `Mi` is the number of detections in that epoch and may be 0..NPLANET. All IMG detections are treated as confirmed companion relative astrometry. Within one epoch, two detections cannot be assigned to the same companion.\n\n")
    lines.extend([
        "## Parameter order\n\n",
    ])
    for i, name in enumerate(names):
        lines.append(f"{i}. `{name}`\n")
    lines.extend([
        "\n## Orbital convention\n\n",
        "The sampled `omega_planet_deg` is always the argument of periastron of the companion orbit relative to the host star. IMG uses this angle directly. RV, AST, and AL use the corresponding stellar reflex direction internally.\n\n",
        "The sampled phase is `tau_frac`: `M(t_ref) = 2*pi*tau_frac` and `M(t) = 2*pi*tau_frac + 2*pi*(t - t_ref)/P`. No sampled `T0` or `M0` is used.\n\n",
        "Two-dimensional AST and IMG offsets use `Delta alpha* = Delta alpha cos(delta)`.\n",
    ])
    if spec.has_al and spec.al_profile_dim == 5:
        lines.append("\nAL-only profiles out `dalpha0*`, `ddelta0`, `pmra*`, `pmdec`, and `parallax` inside the likelihood. These five quantities are not sampled.\n")
    elif spec.has_al and spec.al_profile_dim == 4:
        lines.append("\nThe AL branch profiles out `dalpha0*`, `ddelta0`, `pmra*`, and `pmdec`. The parallax is sampled because it is also used by the companion orbit.\n")
    (outdir / "README.md").write_text("".join(lines), encoding="utf-8")


def copy_share_files(share_dir: Path, outdir: Path) -> None:
    if not share_dir.exists():
        raise SystemExit(f"share directory not found: {share_dir}")
    if not share_dir.is_dir():
        raise SystemExit(f"share path is not a directory: {share_dir}")
    for src in share_dir.iterdir():
        dst = outdir / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, dst)


def generate_one(model: str, n_companions: int, outdir: Path, n_rv_sources: int, copy_share: bool, share_dir: Path) -> None:
    spec = SPECS[model]
    outdir.mkdir(parents=True, exist_ok=True)
    if copy_share:
        copy_share_files(share_dir, outdir)
    (outdir / "user_logll.c").write_text(TEMPLATE_FUNCS[model](n_companions, n_rv_sources), encoding="utf-8")
    write_input_ini(outdir, spec, n_companions, n_rv_sources)
    write_readme(outdir, spec, n_companions, n_rv_sources)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GLEAM model files.")
    parser.add_argument("--model", choices=MODEL_NAMES, help="Model branch to generate.")
    parser.add_argument("--all", action="store_true", help="Generate all supported branches.")
    parser.add_argument("--n-companions", type=int, required=True, help="Number of Keplerian companions.")
    parser.add_argument("--n-rv-sources", type=int, default=1, help="Number of RV instruments used for RV branches.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--copy-share", action="store_true", help="Copy every file and subdirectory from ./share into each generated model directory.")
    parser.add_argument("--share-dir", default="share", help="Shared-source directory used with --copy-share. Default: ./share")
    args = parser.parse_args()

    if args.n_companions < 1:
        raise SystemExit("--n-companions must be positive")
    if args.n_rv_sources < 1:
        raise SystemExit("--n-rv-sources must be positive")
    if not args.all and args.model is None:
        raise SystemExit("Use --model or --all")

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    share_dir = Path(args.share_dir)

    if args.all:
        for model in MODEL_NAMES:
            generate_one(model, args.n_companions, root / model, args.n_rv_sources, args.copy_share, share_dir)
    else:
        generate_one(args.model, args.n_companions, root, args.n_rv_sources, args.copy_share, share_dir)


if __name__ == "__main__":
    main()

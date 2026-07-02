#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "alloc.h"
#include "sofa.h"
#include "sofam.h"

extern int ndim_data;
extern int N_parm;
extern double Mstar_Msun;
extern double calc_M_tau(double t_jd, double t_ref_jd, double period_days, double tau_frac);
extern double newton_solver(double M, double e);
extern double calc_nv(double EE, double ecc);
extern double calc_RV(double K, double nv, double omega, double ecc);
extern double period_mass_to_au(double period_days, double mtot_Msun);
extern double alpha_star_mas_from_K(double K_mps, double period_days, double ecc, double cosi, double plx_mas);
extern double solve_mp_mjup_from_K(double K_mps, double period_days, double ecc, double cosi, double ms_Msun);
extern int solve_linear_system(int n, double *A, double *b, double *x);

#ifndef NPLANET
#define NPLANET 2
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

typedef struct OrbitPack {
    double P_day;
    double e;
    double amp;
    double tau_frac;
    double omega_planet_deg;
    double cosi;
    double Omega_deg;
    double alpha_rel_mas;
    double alpha_star_mas;
} OrbitPack;

static inline double sqr(double x) { return x * x; }

static double log_normal_1d(double res, double sig)
{
    if (!(sig > 0.0) || !isfinite(sig)) return -INFINITY;
    return -0.5 * log(TWOPI * sig * sig) - 0.5 * sqr(res / sig);
}

static int basic_planet_ok(const OrbitPack *p, int need_sky)
{
    if (!(p->P_day > 0.0) || !isfinite(p->P_day)) return 0;
    if (!(p->e >= 0.0 && p->e < 1.0) || !isfinite(p->e)) return 0;
    if (!(p->amp >= 0.0) || !isfinite(p->amp)) return 0;
    if (need_sky) {
        if (!(p->cosi >= -1.0 && p->cosi <= 1.0) || !isfinite(p->cosi)) return 0;
    }
    return 1;
}

static void thiele_innes(double rho_mas, double omega_deg, double Omega_deg, double cosi,
                         double *A, double *B, double *F, double *G)
{{
    double omega = omega_deg * DEG2RAD;
    double Omega = Omega_deg * DEG2RAD;
    double cO = cos(Omega), sO = sin(Omega);
    double cw = cos(omega), sw = sin(omega);

    *A = rho_mas * ( cw * cO - sw * sO * cosi);
    *B = rho_mas * ( cw * sO + sw * cO * cosi);
    *F = rho_mas * (-sw * cO - cw * sO * cosi);
    *G = rho_mas * (-sw * sO + cw * cO * cosi);
}}

static int sky_offset_from_orbit(double t_jd, double t_ref_jd, const OrbitPack *p,
                                 double scale_mas, int stellar_reflex,
                                 double *dalpha_star_mas, double *ddelta_mas)
{{
    double A, B, F, G, M, E, X, Y, rho;

    M = calc_M_tau(t_jd, t_ref_jd, p->P_day, p->tau_frac);
    E = newton_solver(M, p->e);
    if (!isfinite(E)) return 0;

    X = cos(E) - p->e;
    Y = sqrt(fmax(0.0, 1.0 - p->e * p->e)) * sin(E);
    rho = stellar_reflex ? -scale_mas : scale_mas;
    thiele_innes(rho, p->omega_planet_deg, p->Omega_deg, p->cosi, &A, &B, &F, &G);

    *dalpha_star_mas = B * X + G * Y;
    *ddelta_mas      = A * X + F * Y;
    return isfinite(*dalpha_star_mas) && isfinite(*ddelta_mas);
}}



static double img_assignment_rec(int d, int m, int nplanet, const double *logp, int *used)
{{
    int k;
    double best = -INFINITY;
    double vals[MAX_PLANET];
    int nv = 0;

    if (d == m) return 0.0;

    for (k = 0; k < nplanet; ++k) {{
        if (!used[k]) {{
            double v;
            used[k] = 1;
            v = logp[d * nplanet + k] + img_assignment_rec(d + 1, m, nplanet, logp, used);
            used[k] = 0;
            vals[nv++] = v;
            if (v > best) best = v;
        }}
    }}

    if (!isfinite(best)) return -INFINITY;
    {{
        double s = 0.0;
        for (k = 0; k < nv; ++k) s += exp(vals[k] - best);
        return best + log(s);
    }}
}}

static double img_epoch_logsum(const double *logp, int m, int nplanet)
{{
    int used[MAX_PLANET];
    int k;
    if (m < 0 || m > nplanet || nplanet > MAX_PLANET) return -INFINITY;
    if (m == 0) return 0.0;
    for (k = 0; k < MAX_PLANET; ++k) used[k] = 0;
    return img_assignment_rec(0, m, nplanet, logp, used);
}}


double logll_beta(double *ptr_one_chain,
                  int nline_data,
                  double *data_NlineNdim,
                  double beta_one)
{
    OrbitPack pl[MAX_PLANET];
    int ip, i, a, b, ie, img_row;
    int expected_N_parm = 17;
    double t_ref_jd = 0.0;
    double logll_AL = 0.0;
    double logll_IMG = 0.0;

    int iStart_AL = 0, N_al = 0;
    int iStart_IMG = 0, N_img_epoch = 0;

    double plx_mas = 0.0;
    double sigma_jit_AL = 0.0;
    double sigma_jit_IMG = 0.0;

    int n_used = 0;
    double normal[64], rhs[8], x_lin[8];

    if (NPLANET > MAX_PLANET) return -INFINITY;
    if (N_parm != expected_N_parm) return -INFINITY;

    for (ip = 0; ip < NPLANET; ++ip) {
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
    plx_mas = ptr_one_chain[14];
    sigma_jit_AL = ptr_one_chain[15];
    if (!(sigma_jit_AL >= 0.0) || !isfinite(sigma_jit_AL)) return -INFINITY;
    sigma_jit_IMG = ptr_one_chain[16];
    if (!(sigma_jit_IMG >= 0.0) || !isfinite(sigma_jit_IMG)) return -INFINITY;
    if (!(Mstar_Msun > 0.0) || !isfinite(Mstar_Msun)) return -INFINITY;
    if (!(plx_mas > 0.0) || !isfinite(plx_mas)) return -INFINITY;
    for (ip = 0; ip < NPLANET; ++ip) {
        double mp_sun = pl[ip].amp * MJUP_TO_MSUN;
        double a_rel_AU = period_mass_to_au(pl[ip].P_day, Mstar_Msun + mp_sun);
        pl[ip].alpha_rel_mas = a_rel_AU * plx_mas;
        pl[ip].alpha_star_mas = pl[ip].alpha_rel_mas * mp_sun / (Mstar_Msun + mp_sun);
        if (!isfinite(pl[ip].alpha_rel_mas) || !isfinite(pl[ip].alpha_star_mas)) return -INFINITY;
    }
    /* AL rows and reference epoch. */
    iStart_AL = (int)floor(DAT(0, 0) + 0.5);
    iStart_IMG = (int)floor(DAT(0, 1) + 0.5);
    N_img_epoch = (int)floor(DAT(0, 2) + 0.5);
    t_ref_jd = DAT(0, 3) + DAT(0, 4);
    N_al = iStart_IMG - iStart_AL;
    if (iStart_AL < 1 || N_al < 4 || iStart_AL + N_al > nline_data) return -INFINITY;

    for (a = 0; a < 4; ++a) {
        rhs[a] = 0.0;
        x_lin[a] = 0.0;
        for (b = 0; b < 4; ++b) normal[a * 4 + b] = 0.0;
    }

    for (i = 0; i < N_al; ++i) {
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

        for (ip = 0; ip < NPLANET; ++ip) {
            double da, dd;
            if (!sky_offset_from_orbit(t_jd, t_ref_jd, &pl[ip], pl[ip].alpha_star_mas, 1, &da, &dd)) return -INFINITY;
            da_sum += da;
            dd_sum += dd;
        }
        al_orbit_mas = da_sum * sin_psi + dd_sum * cos_psi;
        dt_yr = (t_jd - t_ref_jd) / 365.25;
        H[0] = sin_psi;
        H[1] = cos_psi;
        H[2] = dt_yr * sin_psi;
        H[3] = dt_yr * cos_psi;
        yprime = al_obs_mas - al_orbit_mas - plx_mas * p_al;
        sig_eff = sqrt(sigma_mas * sigma_mas + sigma_jit_AL * sigma_jit_AL);
        if (!(sig_eff > 0.0) || !isfinite(sig_eff)) return -INFINITY;
        w = 1.0 / (sig_eff * sig_eff);

        for (a = 0; a < 4; ++a) {
            rhs[a] += H[a] * w * yprime;
            for (b = 0; b < 4; ++b) normal[a * 4 + b] += H[a] * w * H[b];
        }
        n_used++;
    }

    if (n_used < 4) return -INFINITY;
    if (!solve_linear_system(4, normal, rhs, x_lin)) return -INFINITY;

    for (i = 0; i < N_al; ++i) {
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
        for (ip = 0; ip < NPLANET; ++ip) {
            double da, dd;
            if (!sky_offset_from_orbit(t_jd, t_ref_jd, &pl[ip], pl[ip].alpha_star_mas, 1, &da, &dd)) return -INFINITY;
            da_sum += da;
            dd_sum += dd;
        }
        al_orbit_mas = da_sum * sin_psi + dd_sum * cos_psi;
        dt_yr = (t_jd - t_ref_jd) / 365.25;
        al_5p_mas = x_lin[0] * sin_psi
                  + x_lin[1] * cos_psi
                  + x_lin[2] * dt_yr * sin_psi
                  + x_lin[3] * dt_yr * cos_psi
                  + plx_mas * p_al;
        al_model_mas = al_5p_mas + al_orbit_mas;
        sig_eff = sqrt(sigma_mas * sigma_mas + sigma_jit_AL * sigma_jit_AL);
        lp = log_normal_1d(al_obs_mas - al_model_mas, sig_eff);
        if (!isfinite(lp)) return -INFINITY;
        logll_AL += lp;
    }
    /* IMG rows. Confirmed companion relative astrometry. */
    if (N_img_epoch < 0) return -INFINITY;
    img_row = iStart_IMG;
    for (ie = 0; ie < N_img_epoch; ++ie) {
        int m = (int)floor(DAT(0, 5 + ie) + 0.5);
        double logp[MAX_PLANET * MAX_PLANET];
        if (m < 0 || m > NPLANET || NPLANET > MAX_PLANET) return -INFINITY;
        if (img_row + m > nline_data) return -INFINITY;

        for (i = 0; i < m; ++i) {
            int row = img_row + i;
            double tJD = DAT(row, 0);
            double da_obs = DAT(row, 1);
            double dd_obs = DAT(row, 2);
            double sig_a0 = DAT(row, 3);
            double sig_d0 = DAT(row, 4);
            double sig_a = sqrt(sig_a0 * sig_a0 + (sigma_jit_IMG) * (sigma_jit_IMG));
            double sig_d = sqrt(sig_d0 * sig_d0 + (sigma_jit_IMG) * (sigma_jit_IMG));
            int jp;
            if (!(sig_a > 0.0) || !(sig_d > 0.0)) return -INFINITY;

            for (jp = 0; jp < NPLANET; ++jp) {
                double da_model, dd_model;
                double lp;
                if (!sky_offset_from_orbit(tJD, t_ref_jd, &pl[jp], pl[jp].alpha_rel_mas, 0, &da_model, &dd_model)) return -INFINITY;
                lp = log_normal_1d(da_obs - da_model, sig_a) + log_normal_1d(dd_obs - dd_model, sig_d);
                logp[i * NPLANET + jp] = lp;
            }
        }
        logll_IMG += img_epoch_logsum(logp, m, NPLANET);
        if (!isfinite(logll_IMG)) return -INFINITY;
        img_row += m;
    }
    if (img_row != nline_data) return -INFINITY;

    return (logll_AL + logll_IMG) * beta_one;
}

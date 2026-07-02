#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "alloc.h"

#ifndef PI
#define PI 3.141592653589793238462643383279502884
#endif
#ifndef TWOPI
#define TWOPI (2.0 * PI)
#endif

#define MSUN_CGS 1.9884e33
#define MJUP_CGS 1.8986e30
#define AU_CGS   1.495978707e13
#define GG_CGS   6.67259e-8

static double wrap_twopi(double x)
{
    x = fmod(x, TWOPI);
    if (x < 0.0) x += TWOPI;
    return x;
}

double calc_nv(double EE, double ecc)
{
    return 2.0 * atan2(sqrt(1.0 + ecc) * sin(EE / 2.0),
                       sqrt(1.0 - ecc) * cos(EE / 2.0));
}

/* Legacy T0 convention, kept for old models. */
double calc_M(double t, double period_days, double T0)
{
    return wrap_twopi(TWOPI * (t - T0) / period_days);
}

/* tau_frac convention used by the GLEAM templates. */
double calc_M_tau(double t_jd, double t_ref_jd, double period_days, double tau_frac)
{
    return wrap_twopi(TWOPI * tau_frac + TWOPI * (t_jd - t_ref_jd) / period_days);
}

double calc_RV(double K1, double nv, double omega, double ecc)
{
    return K1 * (cos(nv + omega) + ecc * cos(omega));
}

double calc_K1(double ms_Sun, double mp_Jup, double sin_i, double ecc, double period_days)
{
    double period_yr = period_days / 365.25;
    double mp_sun = mp_Jup * (MJUP_CGS / MSUN_CGS);
    double fac = sqrt(fmax(1.0e-300, 1.0 - ecc * ecc));
    return 28.4329 / fac * (mp_Jup * sin_i)
         * pow(ms_Sun + mp_sun, -2.0 / 3.0)
         * pow(period_yr, -1.0 / 3.0);
}

double newton_solver(double M, double e)
{
    double E, dE;
    int it;

    M = wrap_twopi(M);
    E = (e < 0.8) ? M : PI;

    for (it = 0; it < 100; ++it) {
        double f  = E - e * sin(E) - M;
        double fp = 1.0 - e * cos(E);
        if (!isfinite(f) || !isfinite(fp) || fabs(fp) < 1.0e-15) return NAN;
        dE = -f / fp;
        E += dE;
        if (fabs(dE) < 1.0e-13) return E;
    }

    return E;
}

double period_to_au(double period_days, double ms_Msun)
{
    double period_second = period_days * 86400.0;
    double a = period_second / TWOPI;
    double r = a * a * GG_CGS * MSUN_CGS * ms_Msun;
    return pow(r, 1.0 / 3.0) / AU_CGS;
}

double period_mass_to_au(double period_days, double mtot_Msun)
{
    double period_yr = period_days / 365.25;
    return pow(period_yr * period_yr * mtot_Msun, 1.0 / 3.0);
}

double safe_sini_from_cosi(double cosi)
{
    double s2 = 1.0 - cosi * cosi;
    if (s2 <= 0.0) return 1.0e-12;
    return sqrt(s2);
}

double alpha_star_mas_from_K(double K_mps, double period_days, double ecc,
                             double cosi, double plx_mas)
{
    double P_sec = period_days * 86400.0;
    double sini = safe_sini_from_cosi(cosi);
    double fac = sqrt(fmax(0.0, 1.0 - ecc * ecc));
    double a_star_m = K_mps * P_sec * fac / (TWOPI * sini);
    double a_star_AU = a_star_m / (AU_CGS * 1.0e-2);
    return a_star_AU * plx_mas;
}

static double K_from_mp_local(double mp_Mjup, double ms_Msun,
                              double sin_i, double ecc, double period_days)
{
    return calc_K1(ms_Msun, mp_Mjup, sin_i, ecc, period_days);
}

double solve_mp_mjup_from_K(double K_mps, double period_days, double ecc,
                            double cosi, double ms_Msun)
{
    double sini, lo, hi, mid, kmid;
    int it;

    if (!(K_mps >= 0.0) || !(period_days > 0.0) || !(ecc >= 0.0 && ecc < 1.0) ||
        !(ms_Msun > 0.0) || !(cosi >= -1.0 && cosi <= 1.0)) return NAN;

    sini = safe_sini_from_cosi(cosi);
    if (K_mps == 0.0) return 0.0;

    lo = 0.0;
    hi = 1.0;
    while (hi < 1.0e6 && K_from_mp_local(hi, ms_Msun, sini, ecc, period_days) < K_mps) {
        hi *= 2.0;
    }
    if (hi >= 1.0e6) return NAN;

    for (it = 0; it < 80; ++it) {
        mid = 0.5 * (lo + hi);
        kmid = K_from_mp_local(mid, ms_Msun, sini, ecc, period_days);
        if (kmid < K_mps) lo = mid;
        else hi = mid;
    }

    return 0.5 * (lo + hi);
}

double *read_1d_data(char *path, int nline, char *delimiter)
{
    FILE *file = fopen(path, "r");
    double *data;
    char buffer[1024];
    char *token;
    int i = 0;

    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return NULL;
    }

    data = alloc_1d_double(nline);
    while (fgets(buffer, 1024, file) && i < nline) {
        token = strtok(buffer, delimiter);
        if (token != NULL) data[i++] = atof(token);
    }

    fclose(file);
    return data;
}

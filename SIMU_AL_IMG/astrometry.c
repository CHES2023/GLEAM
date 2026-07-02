#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "alloc.h"
#include "sofa.h"
#include "sofam.h"

#ifndef PI
#define PI 3.141592653589793238462643383279502884
#endif

/* Basic helper kept for older astrometric models. */
double calc_osi(double mp_Mearth, double ms_Msun, double a_AU, double d_pc)
{
    return 3.0 * mp_Mearth * pow(ms_Msun, -1.0) * a_AU * pow(d_pc, -1.0);
}

int solve_linear_system(int n, double *A, double *b, double *x)
{
    int i, j, k, piv;
    double maxabs;
    double aug[8][9];

    if (n <= 0 || n > 8) return 0;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) aug[i][j] = A[i*n + j];
        aug[i][n] = b[i];
        x[i] = 0.0;
    }

    for (k = 0; k < n; ++k) {
        piv = k;
        maxabs = fabs(aug[k][k]);
        for (i = k + 1; i < n; ++i) {
            double v = fabs(aug[i][k]);
            if (v > maxabs) {
                maxabs = v;
                piv = i;
            }
        }
        if (!(maxabs > 1.0e-20) || !isfinite(maxabs)) return 0;

        if (piv != k) {
            for (j = k; j <= n; ++j) {
                double tmp = aug[k][j];
                aug[k][j] = aug[piv][j];
                aug[piv][j] = tmp;
            }
        }

        for (i = k + 1; i < n; ++i) {
            double f = aug[i][k] / aug[k][k];
            aug[i][k] = 0.0;
            for (j = k + 1; j <= n; ++j) aug[i][j] -= f * aug[k][j];
        }
    }

    for (i = n - 1; i >= 0; --i) {
        double s = aug[i][n];
        for (j = i + 1; j < n; ++j) s -= aug[i][j] * x[j];
        if (!(fabs(aug[i][i]) > 1.0e-20) || !isfinite(aug[i][i])) return 0;
        x[i] = s / aug[i][i];
        if (!isfinite(x[i])) return 0;
    }

    return 1;
}

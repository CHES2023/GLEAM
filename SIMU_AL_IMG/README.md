# GLEAM AL_IMG model

Compiled companion count: `2`.

## Row 0

col0 AL_start_row, col1 IMG_start_row, col2 N_img_epoch, col3 ep1a, col4 ep1b, col5... Mi for each IMG epoch.

The reference epoch is `t_ref = ep1a + ep1b`.

## Data rows

- AL rows: col0 tJD, col1 AL_mas, col2 AL_err_mas, col3 scanAngle_rad, col4 sin_psi, col5 cos_psi, col6 parallaxFactorAL, col7 outlier_flag, remaining columns optional.
- IMG rows: col0 tJD, col1 Delta_alpha*_mas, col2 Delta_delta_mas, col3 sigma_alpha*_mas, col4 sigma_delta_mas. The offset is companion relative to host star.

For IMG epochs, `Mi` is the number of detections in that epoch and may be 0..NPLANET. All IMG detections are treated as confirmed companion relative astrometry. Within one epoch, two detections cannot be assigned to the same companion.

## Parameter order

0. `P1_days`
1. `e1`
2. `Mp_Mjup1`
3. `tau1_frac`
4. `omega1_planet_deg`
5. `cosi1`
6. `Omega1_deg`
7. `P2_days`
8. `e2`
9. `Mp_Mjup2`
10. `tau2_frac`
11. `omega2_planet_deg`
12. `cosi2`
13. `Omega2_deg`
14. `plx_mas`
15. `sigma_jit_AL_mas`
16. `sigma_jit_IMG_mas`

## Orbital convention

The sampled `omega_planet_deg` is always the argument of periastron of the companion orbit relative to the host star. IMG uses this angle directly. RV, AST, and AL use the corresponding stellar reflex direction internally.

The sampled phase is `tau_frac`: `M(t_ref) = 2*pi*tau_frac` and `M(t) = 2*pi*tau_frac + 2*pi*(t - t_ref)/P`. No sampled `T0` or `M0` is used.

Two-dimensional AST and IMG offsets use `Delta alpha* = Delta alpha cos(delta)`.

The AL branch profiles out `dalpha0*`, `ddelta0`, `pmra*`, and `pmdec`. The parallax is sampled because it is also used by the companion orbit.

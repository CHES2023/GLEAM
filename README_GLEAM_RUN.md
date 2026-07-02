# 1. GLEAM model generator

This generator writes the branch-specific files for the GLEAM Keplerian likelihood models:

- `user_logll.c`
- `input.ini`
- `README.md`

Shared source files are not built into the generator. If a project keeps common C files in a local `share/` directory, use `--copy-share` to copy all files and subdirectories from `share/` into each generated model directory.

## Supported branches

```text
AL
AST
RV
AL_IMG
AL_RV
AST_IMG
AST_RV
IMG_RV
```

## Use examples

Generate one branch: 

```bash
python make_gleam.py --model AST_IMG --n-companions 2 --outdir SIMU_AST_IMG
```

Generate one branch and copy all files from `./share` into the output directory:

```bash
python make_gleam.py --model AL_RV --n-companions 2 --n-rv-sources 5 --outdir SIMU_RV --copy-share
```

Generate all branches:

```bash
python make_gleam_fit.py --all --n-companions 2 --n-rv-sources 1 --outdir generated_N2
```

With `--all --copy-share`, the contents of the shared-source directory are copied into each branch subdirectory.

## Conventions

The sampled `omega_planet_deg` is always the argument of periastron of the companion orbit relative to the host star. IMG uses this angle directly. RV, AST, and AL use the corresponding stellar reflex direction internally.

The sampled phase is `tau_frac`:

```text
M(t_ref) = 2*pi*tau_frac
M(t) = 2*pi*tau_frac + 2*pi*(t - t_ref)/P
```

Two-dimensional AST and IMG offsets use `Delta alpha* = Delta alpha cos(delta)`.

 

# 2. Prior generator

This script generates a `user_prior.c` file for the Nii-C fitting code.
It creates parameter ranges, initialization functions, proposal scales, boundary checks, and the final `log_prior` function.

## User edit

Edit these values near the top of `generate_prior.py`:

```python
N_PARAM = 15
OUTPUT_C_FILE = "SIMU_RV/user_prior.c"
DEFAULT_PRIOR = "uniform"
PRIOR_TYPES = {
    0: "trunc_gaussian",
    1: "beta",
    5: "trunc_gaussian",
    6: "beta",
}
```

Main settings:

- `N_PARAM`: total number of sampled parameters, `para0` to `para(N_PARAM-1)`.
- `OUTPUT_C_FILE`: output path of the generated C file.
- `DEFAULT_PRIOR`: prior type for parameters not listed in `PRIOR_TYPES`.
- `PRIOR_TYPES`: prior type for selected parameters.

Supported prior types:

```text
uniform
beta
trunc_gaussian
```

## Examples

Run directly:

```bash
python generate_prior.py
```

Or override settings from the command line:

```bash
python generate_prior.py --n-param 15 --beta 1,6 --trunc-gaussian 0,5 --output SIMU_RV/user_prior.c
```

To force some parameters back to uniform:

```bash
python generate_prior.py --uniform 1,6
```

# 3. Full fitting workflow

A complete GLEAM run usually follows these steps.

## Step 1: Generate the model directory

Use `make_gleam.py` to select the fitting branch and the model dimensions. The script writes the corresponding configuration files into the output directory. If the directory does not exist, it is created automatically.

Example:

```bash
python make_gleam.py --model AST_RV --n-companions 2 --n-rv-sources 1 --outdir SIMU_AST_RV --copy-share
```

The generated directory contains the branch-specific files, such as `user_logll.c`, `input.ini`, and the branch-level `README.md`.

Common shared files should also be placed in the same output directory. This can be done automatically with `--copy-share`, which copies all files and subdirectories from the local `share/` directory. Alternatively, the user may copy these files manually.

## Step 2: Generate the prior file

Use `generate_gleam.py` to generate the `user_prior.c` file with the correct number of fitting parameters for the selected branch.

Example:

```bash
python generate_gleam.py --n-param 21 --output SIMU_AST_RV/user_prior.c
```

The output `user_prior.c` should be written into the same model directory created in Step 1.

## Step 3: Prepare the input files

Follow the branch-level `README.md` generated in Step 1.

Edit `input.ini` according to the selected branch, number of companions, number of RV sources, reference epoch, stellar parameters, and sampling settings.

Create `input.dat` manually and arrange the fitting data in the required order. The exact row and column format depends on the selected branch. Use the generated branch-level `README.md` as the format reference.

## Step 4: Compile and run

Compile the code in the generated model directory:

```bash
make
```

If this is not the first compilation, clean the previous build first:

```bash
make clean
make
```

The executable is `a.out`. Run it with `mpirun` using the desired number of MPI processes.

Example command used on a Linux system:

```bash
mpirun --use-hwthread-cpus --oversubscribe -np 8 ./a.out
```

On a Linux server, it can also be run in the background, for example:

```bash
nohup mpirun -np 8 ./a.out > output.log 2>&1 &
```

In general, the run command has the form:

```bash
mpirun -np <N_processes> ./a.out
```

where `<N_processes>` should normally match the number of temperature chains or MPI ranks used by the fitting setup.

## Step 5: Check the chain files

The sampled chains are saved in the `chains/` directory inside the generated model directory.

With the current temperature-chain setting, `chain7.dat` is the cold-chain output. Each row stores one MCMC step. The first `N_PARAM` columns are the sampled fitting parameters. The last three columns are:

```text
log-likelihood, total step number, total accepted step number
```

These chain files can be used for posterior analysis, convergence checks, and plotting.

## Step 6: Run the example cases

The folders `SIMU_AL_IMG` and `SIMU_AST_RV` provide two complete tested examples, including ready-to-run input files and data. They can be used to check the installation, test the generator output, and compare the expected file formats.


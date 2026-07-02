#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <ctype.h>

#define DEST_SIZE 1024

/* Model-level constants read from input.ini when present. */
double Mstar_Msun = 0.0;
double rv_sys_kms = 0.0;

int N_parm;
int N_iter;
int N_stoptune;
int N_begintune;
int Tune_Ladder;
int N_stopTuneLadder;
double scale_tune_ladder;
double zero_stretch;
int N_beta;
double *Beta_Values;
int n_iter_a_stack;
int n_iter_a_batch_base;
int n_iter_a_batch_rand;
int N_swap;
int Swapmode;
int n_iter_in_tune;
double ar_ok_lower;
double ar_ok_upper;
double ar_best;
double ar_accept_diff;
double sigma_scale_min;
double sigma_scale_max;
double sigma_scale_half_ratio;
double sigma_jumpin_ratio;
unsigned i_save_begin;
double init_gp_ratio;
unsigned init_rand_seed;
int Fout_Len;
char *FoutPre;
char *FoutSuf;
char *results_dir;
char *Data_file;
char *Delimiter;
int ndim_data;

char *read_onepara(char *path, char *para_name);
char *read_onepara_optional(char *path, char *para_name);
int read_input_ini(char *path);
void read_chains_grid(char *path);
int read_beta_values(char *path);
void read_chain_out_name(char *path);
void read_data_desc(char *path);
void read_sampling_para(char *path);
void read_model_ini(char *path);
void read_RV_ini(char *path);
int make_dir(char *path);

static char *trim_left(char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    return s;
}

static void trim_right(char *s)
{
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n-1])) {
        s[n-1] = '\0';
        n--;
    }
}

static char *copy_string(const char *s)
{
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (out) memcpy(out, s, n + 1);
    return out;
}

static char *find_key_line(char *path, char *para_name, int required)
{
    FILE *fp;
    char *line_buf = NULL;
    size_t line_buf_size = 0;
    char *found_line = NULL;
    int found = 0;

    fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file '%s'\n", path);
        return NULL;
    }

    while (getline(&line_buf, &line_buf_size, fp) >= 0) {
        char *p = trim_left(line_buf);
        char *colon;
        char key[DEST_SIZE];
        size_t nkey;

        if (*p == '\0' || *p == '#' || *p == '/') continue;
        colon = strchr(p, ':');
        if (colon == NULL) continue;

        nkey = (size_t)(colon - p);
        if (nkey >= sizeof(key)) nkey = sizeof(key) - 1;
        memcpy(key, p, nkey);
        key[nkey] = '\0';
        trim_right(key);

        if (strcmp(key, para_name) == 0) {
            found++;
            free(found_line);
            found_line = copy_string(p);
        }
    }

    fclose(fp);
    free(line_buf);

    if (found > 1) {
        fprintf(stderr, "Error(readin.c): duplicate definition of '%s'.\n", para_name);
        free(found_line);
        return NULL;
    }
    if (found == 0 && required) {
        fprintf(stderr, "Error(readin.c): '%s' not found.\n", para_name);
        return NULL;
    }

    return found_line;
}

char *read_onepara(char *path, char *para_name)
{
    return find_key_line(path, para_name, 1);
}

char *read_onepara_optional(char *path, char *para_name)
{
    return find_key_line(path, para_name, 0);
}

static void require_line(char *para_line, const char *name)
{
    if (para_line == NULL) {
        fprintf(stderr, "Fatal input error: missing %s.\n", name);
        exit(EXIT_FAILURE);
    }
}

void read_chains_grid(char *path)
{
    char *para_line;
    char dummy[DEST_SIZE] = {0};

    para_line = read_onepara(path, "N_parm");
    require_line(para_line, "N_parm");
    sscanf(para_line, "%[^:]:%d", dummy, &N_parm);
    free(para_line);

    para_line = read_onepara(path, "N_iter");
    require_line(para_line, "N_iter");
    sscanf(para_line, "%[^:]:%d", dummy, &N_iter);
    free(para_line);

    para_line = read_onepara(path, "N_stoptune");
    require_line(para_line, "N_stoptune");
    sscanf(para_line, "%[^:]:%d", dummy, &N_stoptune);
    free(para_line);

    para_line = read_onepara(path, "N_begintune");
    require_line(para_line, "N_begintune");
    sscanf(para_line, "%[^:]:%d", dummy, &N_begintune);
    free(para_line);

    para_line = read_onepara(path, "Tune_Ladder");
    require_line(para_line, "Tune_Ladder");
    sscanf(para_line, "%[^:]:%d", dummy, &Tune_Ladder);
    free(para_line);

    para_line = read_onepara(path, "N_stopTuneLadder");
    require_line(para_line, "N_stopTuneLadder");
    sscanf(para_line, "%[^:]:%d", dummy, &N_stopTuneLadder);
    free(para_line);

    para_line = read_onepara(path, "scale_tune_ladder");
    require_line(para_line, "scale_tune_ladder");
    sscanf(para_line, "%[^:]:%lf", dummy, &scale_tune_ladder);
    free(para_line);

    para_line = read_onepara(path, "zero_stretch");
    require_line(para_line, "zero_stretch");
    sscanf(para_line, "%[^:]:%lf", dummy, &zero_stretch);
    free(para_line);

    para_line = read_onepara(path, "N_beta");
    require_line(para_line, "N_beta");
    sscanf(para_line, "%[^:]:%d", dummy, &N_beta);
    free(para_line);
}

void read_chain_out_name(char *path)
{
    char *para_line;
    char dummy[DEST_SIZE] = {0};

    FoutPre = (char*)malloc(DEST_SIZE);
    FoutSuf = (char*)malloc(DEST_SIZE);
    results_dir = (char*)malloc(DEST_SIZE);

    para_line = read_onepara(path, "Fout_Len");
    require_line(para_line, "Fout_Len");
    sscanf(para_line, "%[^:]:%d", dummy, &Fout_Len);
    free(para_line);

    para_line = read_onepara(path, "FoutPre");
    require_line(para_line, "FoutPre");
    sscanf(para_line, "%[^:]:%s", dummy, FoutPre);
    free(para_line);

    para_line = read_onepara(path, "FoutSuf");
    require_line(para_line, "FoutSuf");
    sscanf(para_line, "%[^:]:%s", dummy, FoutSuf);
    free(para_line);

    para_line = read_onepara(path, "results_dir");
    require_line(para_line, "results_dir");
    sscanf(para_line, "%[^:]:%s", dummy, results_dir);
    free(para_line);
}

void read_data_desc(char *path)
{
    char *para_line;
    char dummy[DEST_SIZE] = {0};

    Data_file = (char*)malloc(DEST_SIZE);
    Delimiter = (char*)malloc(DEST_SIZE);

    para_line = read_onepara(path, "Data_file");
    require_line(para_line, "Data_file");
    sscanf(para_line, "%[^:]:%s", dummy, Data_file);
    free(para_line);

    para_line = read_onepara(path, "ndim_data");
    require_line(para_line, "ndim_data");
    sscanf(para_line, "%[^:]:%d", dummy, &ndim_data);
    free(para_line);

    para_line = read_onepara(path, "Delimiter");
    require_line(para_line, "Delimiter");
    sscanf(para_line, "%[^:]:%s", dummy, Delimiter);
    free(para_line);
    if (strcmp(Delimiter, "blank") == 0) Delimiter = " ";
}

int read_beta_values(char *path)
{
    char *para_line;
    char *token;
    int found = 0;

    Beta_Values = (double *)malloc(sizeof(double) * N_beta);
    para_line = read_onepara(path, "Beta_Values");
    require_line(para_line, "Beta_Values");

    token = strtok(para_line, ":");
    while ((token = strtok(NULL, ",")) != NULL && found < N_beta) {
        Beta_Values[found] = atof(token);
        found++;
    }

    free(para_line);
    if (found == N_beta) return EXIT_SUCCESS;

    free(Beta_Values);
    Beta_Values = NULL;
    return EXIT_FAILURE;
}

void read_sampling_para(char *path)
{
    char *para_line;
    char dummy[DEST_SIZE] = {0};

#define READ_INT(KEY, VAR) do { \
    para_line = read_onepara(path, KEY); require_line(para_line, KEY); \
    sscanf(para_line, "%[^:]:%d", dummy, &(VAR)); free(para_line); \
} while (0)

#define READ_UINT(KEY, VAR) do { \
    para_line = read_onepara(path, KEY); require_line(para_line, KEY); \
    sscanf(para_line, "%[^:]:%u", dummy, &(VAR)); free(para_line); \
} while (0)

#define READ_DBL(KEY, VAR) do { \
    para_line = read_onepara(path, KEY); require_line(para_line, KEY); \
    sscanf(para_line, "%[^:]:%lf", dummy, &(VAR)); free(para_line); \
} while (0)

    READ_DBL("init_gp_ratio", init_gp_ratio);
    READ_UINT("init_rand_seed", init_rand_seed);
    READ_UINT("i_save_begin", i_save_begin);
    READ_INT("n_iter_a_stack", n_iter_a_stack);
    READ_INT("n_iter_a_batch_base", n_iter_a_batch_base);
    READ_INT("n_iter_a_batch_rand", n_iter_a_batch_rand);
    READ_INT("n_iter_in_tune", n_iter_in_tune);
    READ_DBL("ar_ok_lower", ar_ok_lower);
    READ_DBL("ar_ok_upper", ar_ok_upper);
    READ_DBL("ar_best", ar_best);
    READ_DBL("ar_accept_diff", ar_accept_diff);
    READ_DBL("sigma_scale_half_ratio", sigma_scale_half_ratio);
    READ_DBL("sigma_scale_min", sigma_scale_min);
    READ_DBL("sigma_scale_max", sigma_scale_max);
    READ_DBL("sigma_jumpin_ratio", sigma_jumpin_ratio);
    READ_INT("N_swap", N_swap);
    READ_INT("Swapmode", Swapmode);

#undef READ_INT
#undef READ_UINT
#undef READ_DBL
}

void read_model_ini(char *path)
{
    char *para_line;
    char dummy[DEST_SIZE] = {0};

    para_line = read_onepara_optional(path, "Mstar_Msun");
    if (para_line != NULL) {
        sscanf(para_line, "%[^:]:%lf", dummy, &Mstar_Msun);
        free(para_line);
    }

    para_line = read_onepara_optional(path, "rv_sys_kms");
    if (para_line != NULL) {
        sscanf(para_line, "%[^:]:%lf", dummy, &rv_sys_kms);
        free(para_line);
    }
}

void read_RV_ini(char *path)
{
    read_model_ini(path);
}

int make_dir(char *path)
{
    if (mkdir(path, 0755) == -1) {
        printf("\nWARNING: dir: %s exists.\n", path);
        printf("Existing chain files may be overwritten.\n\n");
        return 1;
    }
    return 0;
}

int read_input_ini(char *path)
{
    read_chains_grid(path);
    read_chain_out_name(path);
    read_beta_values(path);
    read_data_desc(path);
    read_sampling_para(path);
    read_model_ini(path);
    return 0;
}

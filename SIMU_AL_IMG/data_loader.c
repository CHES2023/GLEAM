/////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"


int get_nlines_of_file(char *path);

int read_data(char *path, double *data_NlineNdim, int nline_data, int ndim_data, char *delimiter);

// load data 
void mpi_data_loader(int my_rank, int root_rank, int nline_data, int ndim_data, double *data_NlineNdim, char *path, char *delimiter)
{
    //
    if( my_rank == root_rank )
    {
        // read data
        read_data(path, data_NlineNdim, nline_data, ndim_data, delimiter);
    } 
    // 
    // MPI_Bcast the user_data
    MPI_Bcast(&data_NlineNdim[0], nline_data*ndim_data, MPI_DOUBLE, root_rank, MPI_COMM_WORLD);
}



int read_data(char *path, double *data_NlineNdim, int nline_data, int ndim_data, char *delimiter)
{
    FILE *fp = fopen(path, "r");
    //
    if (fp == NULL)
    {
        fprintf(stderr, "Error opening file '%s'\n", path);
    }
    // 
    // Read lines using POSIX function getline
    char *line = NULL;
    size_t len = 0;
    // 
    int iline_local;
    iline_local = 0;
    //
    while (iline_local < nline_data && getline(&line, &len, fp) != -1)
    {
        char *token = strtok(line, delimiter);
        if (!token) continue; // 或者报错退出
        // read the first element
        // /// todo to do todo
        data_NlineNdim[iline_local*ndim_data+0] = atof(token);
        // read the other elements
        for(int j=1; j<ndim_data; j++)
        {
            token = strtok(NULL, delimiter);
            data_NlineNdim[iline_local*ndim_data+j] = atof(token);
        }
        iline_local++;
    }
    //
    fclose(fp);
    free(line);     
    //
    if (nline_data == iline_local) 
    { 
        return 0;
    } 
    else
    {
        return 1;
    }
    //
}






// read the number of line by root_rank and broadcast it
int mpi_get_nlines(int nline_data, int my_rank, char *path, int root_rank)
{
    //
    if( my_rank == root_rank )
    {
        // get the number of line of user_data
        nline_data = get_nlines_of_file(path);
    } 
    // 
    // MPI_Bcast the line_number
    MPI_Bcast(&nline_data, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    //
    return nline_data;
}




int get_nlines_of_file(char *path)
{
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error opening file '%s'\n", path);
        return -1;
    }

    int line_number = 0;
    int c;
    int last = '\n';  // 记录最后一个字符（初始化成换行，便于空文件处理）
    int seen_any = 0;

    while ((c = getc(fp)) != EOF) {
        seen_any = 1;
        last = c;
        if (c == '\n') line_number++;
    }

    fclose(fp);

    // 如果文件非空且最后一个字符不是 '\n'，说明最后一行没有换行符，需要补 1 行
    if (seen_any && last != '\n') line_number++;

    return line_number;
}




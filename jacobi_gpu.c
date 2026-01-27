// jacobi.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <mpi.h>
#include <omp.h>

static int file_exists(const char *p) { struct stat st; return stat(p,&st)==0; }

int main(int argc, char **argv) {
    // --- MPI init with threading ---
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        if (0==0) fprintf(stderr, "MPI needs MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc != 3 && argc != 4 && argc != 6) {
    if (rank == 0) {
        fprintf(stderr,
            "Usage: %s <N> <iters> [num_threads] [num_teams threads_per_team]\n",
            argv[0]);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
}

 int N     = atoi(argv[1]);
 int iters = atoi(argv[2]);

int nthreads = (argc >= 4) ? atoi(argv[3]) : omp_get_max_threads();
int teams = (argc == 6) ? atoi(argv[4]) : 0;
int threads_per_team = (argc == 6) ? atoi(argv[5]) : 0;

if (N <= 0 || iters <= 0 || nthreads <= 0) {
    if (rank == 0) fprintf(stderr, "N, iters, num_threads must be positive.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
}

if (argc == 6 && (teams <= 0 || threads_per_team <= 0)) {
    if (rank == 0) fprintf(stderr, "num_teams and threads_per_team must be positive.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
}

    omp_set_num_threads(nthreads);

    // Pick OpenMP device per rank (works with srun --gpus-per-task=1 or mpirun)
    int ngpus = omp_get_num_devices();
    if (ngpus <= 0) {
        if (rank == 0) fprintf(stderr, "No OpenMP target devices. Set OMP_TARGET_OFFLOAD=MANDATORY to catch this.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    omp_set_default_device(rank % ngpus);

    if (rank == 0) printf("GPUS= %d, MPI ranks=%d, OMP threads/rank=%d, N=%d, iters=%d\n", ngpus, size, nthreads, N, iters);

    // --- 1D row decomposition ---
    const int rows_per   = N / size;
    const int extra_rows = N % size;
    const int local_rows = rows_per + (rank < extra_rows ? 1 : 0);

    // global starting interior row for this rank
    const int start_row  = rank * rows_per + (rank < extra_rows ? rank : extra_rows);

    // include ghost rows
    const int total_rows = local_rows + 2;
    const int total_cols = N + 2;
    const int total_size = total_rows * total_cols;

    // --- allocate on host (MPI_Alloc_mem) ---
    MPI_Aint bytes = (MPI_Aint)sizeof(double) * (MPI_Aint)total_size;
    double *mat = NULL, *mat_new = NULL;
    MPI_Alloc_mem(bytes, MPI_INFO_NULL, &mat);
    MPI_Alloc_mem(bytes, MPI_INFO_NULL, &mat_new);

    // --- timing ---
    MPI_Barrier(MPI_COMM_WORLD);
    double init_time = 0.0, comm_time = 0.0, comp_time = 0.0;

    // --- initialize on host ---
    double t0 = MPI_Wtime();
    memset(mat,     0, (size_t)bytes);
    memset(mat_new, 0, (size_t)bytes);

    // interior init
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= local_rows; ++i) {
        for (int j = 1; j <= N; ++j) {
            mat[i*total_cols + j] = 0.5;
        }
    }
    // boundary init
    double incr = 100.0 / (N + 1);
    #pragma omp parallel for
    for (int i = 1; i <= local_rows; ++i) {
        double v = (start_row + i) * incr;
        mat    [i*total_cols + 0] = v;
        mat_new[i*total_cols + 0] = v;
    }
    if (rank == size - 1) {
        int brow = local_rows + 1;
        #pragma omp parallel for
        for (int j = 1; j <= N+1; ++j) {
            double v = j * incr;
            int col = (N + 1) - j;
            mat    [brow*total_cols + col] = v;
            mat_new[brow*total_cols + col] = v;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    init_time = MPI_Wtime() - t0;

    // --- neighbors ---
    const int above = (rank > 0      ? rank - 1 : MPI_PROC_NULL);
    const int below = (rank < size-1 ? rank + 1 : MPI_PROC_NULL);

    // --- RMA windows on HOST memory (expose first/last interior rows) ---
    MPI_Win win_top = MPI_WIN_NULL;    // exposes row 1
    MPI_Win win_bot = MPI_WIN_NULL;    // exposes row local_rows
    MPI_Win_create(&mat[1 * total_cols],
                   (MPI_Aint)(total_cols * sizeof(double)),
                   (int)sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_top);
    MPI_Win_create(&mat[local_rows * total_cols],
                   (MPI_Aint)(total_cols * sizeof(double)),
                   (int)sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_bot);

    // --- OpenMP target mapping (both arrays ping-ponged) ---
    #pragma omp target data map(tofrom: mat[0:total_size], mat_new[0:total_size])
    {
        for (int it = 0; it < iters; ++it) {

            // 1) bring interior rows to HOST (row 1, row local_rows)
            double tc = MPI_Wtime();

            #pragma omp target update from( mat[1 * total_cols          : total_cols] )
            #pragma omp target update from( mat[local_rows * total_cols : total_cols] )

            // 2) RMA GET: neighbors' interior rows -> my ghost rows (HOST)
            MPI_Win_fence(0, win_top);
            MPI_Win_fence(0, win_bot);

            if (below != MPI_PROC_NULL) {
                // get below's row 1 into my bottom ghost
                MPI_Get(&mat[(local_rows + 1) * total_cols], (int)total_cols, MPI_DOUBLE,
                        below, /*disp*/0, (int)total_cols, MPI_DOUBLE, win_top);
            }
            if (above != MPI_PROC_NULL) {
                // get above's last interior row into my top ghost
                MPI_Get(&mat[0 * total_cols], (int)total_cols, MPI_DOUBLE,
                        above, /*disp*/0, (int)total_cols, MPI_DOUBLE, win_bot);
            }

            MPI_Win_fence(0, win_top);
            MPI_Win_fence(0, win_bot);

            // 3) push the updated ghost rows to DEVICE
            #pragma omp target update to( mat[0 * total_cols                : total_cols] )
            #pragma omp target update to( mat[(local_rows + 1) * total_cols : total_cols] )

            comm_time += MPI_Wtime() - tc;

            // 4) GPU compute (one kernel, no copy-back kernel)
            double tp = MPI_Wtime();
            #pragma omp target teams distribute parallel for collapse(2) thread_limit(threads_per_team) num_teams(teams)
            for (int i = 1; i <= local_rows; ++i) {
                for (int j = 1; j <= N; ++j) {
                    mat_new[i*total_cols + j] = 0.25 * (
                        mat[(i-1)*total_cols + j] +
                        mat[(i+1)*total_cols + j] +
                        mat[i*total_cols + (j-1)] +
                        mat[i*total_cols + (j+1)]
                    );
                }
            }

            // 5) ping-pong
            double *tmp = mat; mat = mat_new; mat_new = tmp;

	    comp_time += MPI_Wtime() - tp;
        }
    } // end target data
    
    double total_wall =init_time + comm_time + comp_time;

    // free windows
    MPI_Win_free(&win_top);
    MPI_Win_free(&win_bot);

    // --- timings ---
    //double total_wall = MPI_Wtime() - prog_t0;
    
    double init_max, comm_max, comp_max;
    MPI_Reduce(&init_time, &init_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &comp_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double total_max = init_max + comm_max + comp_max;
    if (rank == 0) {
        printf("Init (max): %.6f  Comm (max): %.6f  Comp (max): %.6f\n Total (max): %.6f\n",
         init_max, comm_max, comp_max, total_max);
    }

    if (rank == 0) {
       // printf("Initialization:  %.6f s\n", init_time);
       // printf("Communication:   %.6f s\n", comm_time);
       // printf("Computation:     %.6f s\n", comp_time);
       // printf("Total wall:      %.6f s\n", total_wall);
       printf("TIME: %.6f \n", total_wall);
    }

    // --- (optional) write a CSV row per rank (append) ---
    
    if (rank == 0) {
        const char *csv = "times.csv";
        int fresh = !file_exists(csv);
        FILE *fp = fopen(csv, "a");
        if (fp) {
            if (fresh) fprintf(fp, "rank,size,N,iters,init_time,comm_time,comp_time ,total_wall_s\n");
            fprintf(fp, "%d,%d,%d,%d,%.9f,%.9f,%.9f,%.9f\n",
                    rank, size, N, iters, init_time, comm_time, comp_time, total_wall);

            fclose(fp);
            printf("Appended timings to %s\n", csv);
        }
    }
     
    // --- (optional) write solution (global (N+2)x(N+2)) ---
    // Layout: rank 0 writes its top ghost+interior (rows 0..local_rows),
    // middle ranks write only interior (start_row+1 .. start_row+local_rows),
    // last rank writes interior+bottom ghost (start_row+1 .. start_row+local_rows+1).
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, "solution.bin",
                  MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_Offset off_rows, nrows;
    const MPI_Offset rowbytes = (MPI_Offset)total_cols * (MPI_Offset)sizeof(double);

    if (rank == 0) {
        off_rows = 0;
        nrows    = (MPI_Offset)(local_rows + 1);
        MPI_File_write_at_all(fh, off_rows * rowbytes,
                              &mat[0], nrows * total_cols, MPI_DOUBLE, MPI_STATUS_IGNORE);
    } else if (rank == size - 1) {
        off_rows = (MPI_Offset)(start_row + 1);
        nrows    = (MPI_Offset)(local_rows + 1);
        MPI_File_write_at_all(fh, off_rows * rowbytes,
                              &mat[1 * total_cols], nrows * total_cols, MPI_DOUBLE, MPI_STATUS_IGNORE);
    } else {
        off_rows = (MPI_Offset)(start_row + 1);
        nrows    = (MPI_Offset)(local_rows);
        MPI_File_write_at_all(fh, off_rows * rowbytes,
                              &mat[1 * total_cols], nrows * total_cols, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&fh);
    
    // --- cleanup ---
    MPI_Free_mem(mat);
    MPI_Free_mem(mat_new);
    MPI_Finalize();
    return 0;
}


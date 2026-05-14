// Quick experiment: how fast is a raw read() of safetensors files into
// posix_memalign'd buffers? This is the ceiling for any user-space loader.

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <time.h>
#include <pthread.h>
#include <string.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

typedef struct {
    const char *path;
    int use_nocache;
    int use_pread_parallel;
    double elapsed_ms;
    long bytes;
} TaskCtx;

#define CHUNK (32 * 1024 * 1024)  // match mlx ParallelFileReader's 32 MiB

typedef struct {
    int fd;
    char *buf;
    long start;
    long end;
} WorkerArg;

static void* worker_pread(void *arg) {
    WorkerArg *w = (WorkerArg*)arg;
    long off = w->start;
    while (off < w->end) {
        long want = CHUNK < (w->end - off) ? CHUNK : (w->end - off);
        ssize_t r = pread(w->fd, w->buf + off, want, off);
        if (r <= 0) return NULL;
        off += r;
    }
    return NULL;
}

#define WORKERS_PER_FILE 8

static void* read_one_file(void *arg) {
    TaskCtx *t = (TaskCtx*)arg;
    int fd = open(t->path, O_RDONLY);
    if (fd < 0) { perror("open"); return NULL; }
    if (t->use_nocache) {
        fcntl(fd, F_NOCACHE, 1);
    }
    struct stat sb;
    fstat(fd, &sb);
    long size = sb.st_size;
    t->bytes = size;

    void *buf = NULL;
    posix_memalign(&buf, 16384, size);

    double t0 = now_ms();
    // Multi-threaded read of one file (matches what MLX's ParallelFileReader does)
    pthread_t workers[WORKERS_PER_FILE];
    WorkerArg wargs[WORKERS_PER_FILE];
    long chunk = (size + WORKERS_PER_FILE - 1) / WORKERS_PER_FILE;
    for (int i = 0; i < WORKERS_PER_FILE; i++) {
        wargs[i].fd = fd;
        wargs[i].buf = (char*)buf;
        wargs[i].start = i * chunk;
        wargs[i].end = (i + 1) * chunk;
        if (wargs[i].end > size) wargs[i].end = size;
        pthread_create(&workers[i], NULL, worker_pread, &wargs[i]);
    }
    for (int i = 0; i < WORKERS_PER_FILE; i++) {
        pthread_join(workers[i], NULL);
    }
    t->elapsed_ms = now_ms() - t0;
    free(buf);
    close(fd);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <0|1 nocache> <file1> [file2] ...\n", argv[0]);
        return 64;
    }
    int nocache = atoi(argv[1]);
    int nfiles = argc - 2;

    TaskCtx *ctxs = calloc(nfiles, sizeof(TaskCtx));
    pthread_t *th = calloc(nfiles, sizeof(pthread_t));
    for (int i = 0; i < nfiles; i++) {
        ctxs[i].path = argv[2 + i];
        ctxs[i].use_nocache = nocache;
    }

    double t0 = now_ms();
    for (int i = 0; i < nfiles; i++) {
        pthread_create(&th[i], NULL, read_one_file, &ctxs[i]);
    }
    long total_bytes = 0;
    for (int i = 0; i < nfiles; i++) {
        pthread_join(th[i], NULL);
        total_bytes += ctxs[i].bytes;
    }
    double wall_ms = now_ms() - t0;

    for (int i = 0; i < nfiles; i++) {
        double gb = ctxs[i].bytes / 1024.0 / 1024.0 / 1024.0;
        double s = ctxs[i].elapsed_ms / 1000.0;
        fprintf(stdout, "  %s: %.2f GiB in %.3f s = %.2f GiB/s%s\n",
                ctxs[i].path, gb, s, gb/s,
                nocache ? " [F_NOCACHE]" : "");
    }
    double tgb = total_bytes / 1024.0 / 1024.0 / 1024.0;
    fprintf(stdout, "wall: %.3f s, %.2f GiB total = %.2f GiB/s aggregate\n",
            wall_ms/1000.0, tgb, tgb/(wall_ms/1000.0));

    free(ctxs);
    free(th);
    return 0;
}

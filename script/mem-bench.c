// STREAM-style memcpy benchmark to find the actual CPU memory bandwidth
// ceiling on this machine. Compares:
//   (a) memcpy anon→anon (pure RAM bandwidth, no file I/O)
//   (b) memcpy mmap'd file → anon (file-backed source, page cache resident)
//   (c) parallel pread file → anon (what MLX does)
//
// Tells us if the loader is bound by memory bandwidth or by VM/page-cache.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <pthread.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

typedef struct {
    const char *src;
    char *dst;
    size_t size;
} CopyArg;

static void* anon_copy_worker(void *arg) {
    CopyArg *c = (CopyArg*)arg;
    memcpy(c->dst, c->src, c->size);
    return NULL;
}

static double bench_anon_memcpy(size_t total_bytes, int threads) {
    char *src, *dst;
    posix_memalign((void**)&src, 16384, total_bytes);
    posix_memalign((void**)&dst, 16384, total_bytes);
    // Pre-fault by writing
    memset(src, 0x42, total_bytes);
    memset(dst, 0x00, total_bytes);

    pthread_t th[threads];
    CopyArg args[threads];
    size_t chunk = total_bytes / threads;
    for (int i = 0; i < threads; i++) {
        args[i].src = src + i * chunk;
        args[i].dst = dst + i * chunk;
        args[i].size = (i == threads - 1) ? total_bytes - i * chunk : chunk;
    }
    double t0 = now_s();
    for (int i = 0; i < threads; i++)
        pthread_create(&th[i], NULL, anon_copy_worker, &args[i]);
    for (int i = 0; i < threads; i++)
        pthread_join(th[i], NULL);
    double elapsed = now_s() - t0;

    free(src);
    free(dst);
    return elapsed;
}

static double bench_mmap_to_anon(const char *path, int threads) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }
    struct stat sb; fstat(fd, &sb);
    size_t size = sb.st_size;

    void *mp = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (mp == MAP_FAILED) { perror("mmap"); close(fd); return -1; }

    // Force pages resident (kernel won't lazy-fault during the copy)
    volatile unsigned int sink = 0;
    for (size_t i = 0; i < size; i += 16384) sink += ((unsigned char*)mp)[i];
    (void)sink;

    char *dst;
    posix_memalign((void**)&dst, 16384, size);
    memset(dst, 0, size);  // pre-fault dst

    pthread_t th[threads];
    CopyArg args[threads];
    size_t chunk = size / threads;
    for (int i = 0; i < threads; i++) {
        args[i].src = (char*)mp + i * chunk;
        args[i].dst = dst + i * chunk;
        args[i].size = (i == threads - 1) ? size - i * chunk : chunk;
    }
    double t0 = now_s();
    for (int i = 0; i < threads; i++)
        pthread_create(&th[i], NULL, anon_copy_worker, &args[i]);
    for (int i = 0; i < threads; i++)
        pthread_join(th[i], NULL);
    double elapsed = now_s() - t0;

    free(dst);
    munmap(mp, size);
    close(fd);
    return elapsed;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file-for-mmap-test>\n", argv[0]);
        return 64;
    }

    size_t size = 4ULL * 1024 * 1024 * 1024;  // 4 GiB
    printf("Hardware concurrency: %d\n", (int)sysconf(_SC_NPROCESSORS_ONLN));
    printf("Anon memcpy benchmark (%.1f GiB)\n", size/1e9);

    int thread_counts[] = {1, 2, 4, 8, 12, 16};
    int nt = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("\n--- anon→anon memcpy ---\n");
    for (int i = 0; i < nt; i++) {
        int t = thread_counts[i];
        double s = bench_anon_memcpy(size, t);
        printf("  threads=%2d  %.3f s  %.1f GiB/s\n",
            t, s, (size/1024.0/1024/1024)/s);
    }

    printf("\n--- mmap(file)→anon memcpy [%s] ---\n", argv[1]);
    for (int i = 0; i < nt; i++) {
        int t = thread_counts[i];
        double s = bench_mmap_to_anon(argv[1], t);
        struct stat sb; stat(argv[1], &sb);
        printf("  threads=%2d  %.3f s  %.1f GiB/s\n",
            t, s, (sb.st_size/1024.0/1024/1024)/s);
    }

    return 0;
}

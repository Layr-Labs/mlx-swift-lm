// Check whether tensors in a safetensors file are page-aligned (16 KiB).
// Determines whether mmap + zero-copy MLXArray is even possible.
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#define PAGE 16384

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <safetensors-file>\n", argv[0]);
        return 64;
    }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }

    unsigned long long jlen;
    read(fd, &jlen, 8);
    char *json = malloc(jlen + 1);
    read(fd, json, jlen);
    json[jlen] = 0;

    long data_base = 8 + (long)jlen;
    printf("header bytes: 8 + %llu (data starts at offset %ld)\n", jlen, data_base);
    printf("data_base page-aligned: %s\n", (data_base % PAGE == 0) ? "yes" : "no");
    printf("\n");

    // Walk JSON for data_offsets. Cheap hand-parser — looks for `"data_offsets":[N,N]`.
    int aligned = 0, total = 0;
    long min_size = 1L<<60, max_size = 0;
    char *p = json;
    while ((p = strstr(p, "\"data_offsets\""))) {
        p = strchr(p, '[');
        if (!p) break;
        long start, end;
        if (sscanf(p, "[%ld,%ld]", &start, &end) != 2) break;
        long abs = data_base + start;
        long size = end - start;
        total++;
        if (abs % PAGE == 0) aligned++;
        if (size < min_size) min_size = size;
        if (size > max_size) max_size = size;
        p++;
    }
    printf("tensors: %d\n", total);
    printf("page-aligned: %d (%.1f%%)\n", aligned, aligned * 100.0 / total);
    printf("size range: %ld B - %ld B (%.1f GiB)\n",
           min_size, max_size, max_size/1024.0/1024.0/1024.0);

    free(json);
    close(fd);
    return 0;
}

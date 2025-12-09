#ifndef CLAHE_H
#define CLAHE_H

#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32
#define CLIP_LIMIT 4

typedef struct {
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;

// --- Host I/O Functions ---
PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

// --- CPU Implementation ---
PGM_IMG apply_clahe_cpu(PGM_IMG img_in);

// --- GPU Kernels ---
__global__ void histogram_lut_kernel(unsigned char* img, int* all_luts, int w, int h, int grid_w, int clip_limit);
__global__ void render_clahe_kernel(unsigned char* img_in, unsigned char* img_out, int* all_luts, int w, int h, int grid_w, int grid_h);

#endif
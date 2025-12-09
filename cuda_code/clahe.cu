#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "clahe.h"

// ==========================================
// 1. I/O FUNCTIONS
// ==========================================
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    PGM_IMG result;
    int v_max;

    in_file = fopen(path, "rb");
    if (in_file == NULL){
        printf("Error: Input file not found!\n");
        exit(1);
    }
    fscanf(in_file, "%s", sbuf); 
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d",&v_max);
    fgetc(in_file); 

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    fread(result.img, sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n", img.w, img.h);
    fwrite(img.img, sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img) {
    if(img.img) free(img.img);
}

// ==========================================
// 2. CPU IMPLEMENTATION (Reference Code)
// ==========================================
void compute_histogram_cpu(unsigned char* data, int w, int h, int start_x, int start_y, int tile_w, int tile_h, int* lut) {
    int hist[256] = {0};
    int excess = 0, cdf = 0, total_pixels = tile_w * tile_h; 

    for (int y = start_y; y < start_y + tile_h; ++y) {
        for (int x = start_x; x < start_x + tile_w; ++x) {
            if(x < w && y < h) hist[data[y * w + x]]++;
        }
    }

    for (int i = 0; i < 256; ++i) {
        if (hist[i] > CLIP_LIMIT) {
            excess += (hist[i] - CLIP_LIMIT);
            hist[i] = CLIP_LIMIT;
        }
    }

    int avg_inc = excess / 256;
    for (int i = 0; i < 256; ++i) hist[i] += avg_inc;
    
    for (int i = 0; i < 256; ++i) {
        cdf += hist[i];
        int val = (int)((float)cdf * 255.0f / total_pixels + 0.5f);
        if (val > 255) val = 255;
        lut[i] = val;
    }
}

PGM_IMG apply_clahe_cpu(PGM_IMG img_in) {
    PGM_IMG img_out;
    int w = img_in.w;
    int h = img_in.h;
    img_out.w = w; img_out.h = h;
    img_out.img = (unsigned char *)malloc(w * h * sizeof(unsigned char));

    int grid_w = (w + TILE_SIZE - 1) / TILE_SIZE;
    int grid_h = (h + TILE_SIZE - 1) / TILE_SIZE;
    int *all_luts = (int *)malloc(grid_w * grid_h * 256 * sizeof(int));

    // Precompute LUTs
    for (int ty = 0; ty < grid_h; ++ty) {
        for (int tx = 0; tx < grid_w; ++tx) {
            int x_start = tx * TILE_SIZE;
            int y_start = ty * TILE_SIZE;
            int actual_w = (x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE;
            int actual_h = (y_start + TILE_SIZE > h) ? (h - y_start) : TILE_SIZE;
            compute_histogram_cpu(img_in.img, w, h, x_start, y_start, actual_w, actual_h, &all_luts[(ty * grid_w + tx) * 256]);
        }
    }

    // Bilinear Interpolation
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float ty_f = (float)y / TILE_SIZE - 0.5f;
            float tx_f = (float)x / TILE_SIZE - 0.5f;
            int y1 = (int)floor(ty_f), x1 = (int)floor(tx_f);
            int y2 = y1 + 1, x2 = x1 + 1;
            float y_weight = ty_f - y1, x_weight = tx_f - x1;

            if (x1 < 0) x1 = 0; if (x2 >= grid_w) x2 = grid_w - 1;
            if (y1 < 0) y1 = 0; if (y2 >= grid_h) y2 = grid_h - 1;

            int val = img_in.img[y * w + x];
            int tl = all_luts[(y1 * grid_w + x1) * 256 + val];
            int tr = all_luts[(y1 * grid_w + x2) * 256 + val];
            int bl = all_luts[(y2 * grid_w + x1) * 256 + val];
            int br = all_luts[(y2 * grid_w + x2) * 256 + val];

            float top = tl * (1.0f - x_weight) + tr * x_weight;
            float bot = bl * (1.0f - x_weight) + br * x_weight;
            img_out.img[y * w + x] = (unsigned char)(top * (1.0f - y_weight) + bot * y_weight + 0.5f);
        }
    }
    free(all_luts);
    return img_out;
}

// ==========================================
// 3. GPU KERNELS
// ==========================================
__global__ void histogram_lut_kernel(unsigned char* img, int* all_luts, int w, int h, int grid_w, int clip_limit) {
    __shared__ int s_hist[256];
    __shared__ int s_excess[256];
    int tx = blockIdx.x, ty = blockIdx.y;
    int x_start = tx * TILE_SIZE, tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (tid < 256) { s_hist[tid] = 0; s_excess[tid] = 0; }
    __syncthreads();

    int px = x_start + threadIdx.x, py = ty * TILE_SIZE + threadIdx.y;
    if (px < w && py < h) atomicAdd(&s_hist[img[py * w + px]], 1);
    __syncthreads();

    if (tid < 256) {
        if (s_hist[tid] > clip_limit) { s_excess[tid] = s_hist[tid] - clip_limit; s_hist[tid] = clip_limit; }
    }
    __syncthreads();

    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) s_excess[tid] += s_excess[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        int avg_inc = s_excess[0] / 256;
        int cdf = 0;
        int tile_pixels = ((x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE) * ((ty * TILE_SIZE + TILE_SIZE > h) ? (h - ty * TILE_SIZE) : TILE_SIZE);
        int* my_lut = &all_luts[(ty * grid_w + tx) * 256];
        for (int i = 0; i < 256; ++i) {
            s_hist[i] += avg_inc; cdf += s_hist[i];
            int val = (int)((float)cdf * 255.0f / tile_pixels + 0.5f);
            my_lut[i] = (val > 255) ? 255 : val;
        }
    }
}

__global__ void render_clahe_kernel(unsigned char* img_in, unsigned char* img_out, int* all_luts, int w, int h, int grid_w, int grid_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float tx_f = (float)x / TILE_SIZE - 0.5f, ty_f = (float)y / TILE_SIZE - 0.5f;
    int x1 = floorf(tx_f), y1 = floorf(ty_f);
    int x2 = x1 + 1, y2 = y1 + 1;
    float xw = tx_f - x1, yw = ty_f - y1;

    if (x1 < 0) x1 = 0; if (x2 >= grid_w) x2 = grid_w - 1;
    if (y1 < 0) y1 = 0; if (y2 >= grid_h) y2 = grid_h - 1;

    int val = img_in[y * w + x];
    int tl = all_luts[(y1 * grid_w + x1) * 256 + val];
    int tr = all_luts[(y1 * grid_w + x2) * 256 + val];
    int bl = all_luts[(y2 * grid_w + x1) * 256 + val];
    int br = all_luts[(y2 * grid_w + x2) * 256 + val];

    float top = tl * (1.0f - xw) + tr * xw;
    float bot = bl * (1.0f - xw) + br * xw;
    img_out[y * w + x] = (unsigned char)(top * (1.0f - yw) + bot * yw + 0.5f);
}
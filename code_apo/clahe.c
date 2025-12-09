#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "clahe.h"

// Macro για έλεγχο σφαλμάτων CUDA (Απαραίτητο για debugging)
#define GPU_Check(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Constant Memory: Γρήγορη πρόσβαση (cache) για σταθερές που διαβάζουν όλα τα threads
__constant__ int d_CLIP_LIMIT;

/**
 * KERNEL 1: Υπολογισμός Ιστογράμματος & LUT ανά Tile
 * Στρατηγική: 1 CUDA Block ανά Tile εικόνας.
 * Βελτιστοποίηση: Χρήση Shared Memory για αποφυγή Global Atomic collisions.
 */
__global__ void histogram_lut_kernel(unsigned char* img, int* all_luts, int w, int h, int grid_w) {
    // Shared Memory για το ιστόγραμμα (γρήγορη πρόσβαση)
    __shared__ int s_hist[256];
    // Shared Memory για τον υπολογισμό του excess (για το clipping reduction)
    __shared__ int s_excess[256];

    // Συντεταγμένες Tile
    int tx = blockIdx.x;
    int ty = blockIdx.y;
    
    // Αρχικό pixel του Tile
    int x_start = tx * TILE_SIZE;
    int y_start = ty * TILE_SIZE;

    // IDs του thread
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int local_tid = tid_y * blockDim.x + tid_x; // Linear ID (0-1023)

    // --- ΒΗΜΑ 1: Αρχικοποίηση Shared Memory ---
    // Μόνο τα πρώτα 256 threads μηδενίζουν από μία θέση (Coalesced logic)
    if (local_tid < 256) {
        s_hist[local_tid] = 0;
        s_excess[local_tid] = 0;
    }
    __syncthreads(); // Barrier: Περιμένουμε να μηδενιστούν όλα

    // --- ΒΗΜΑ 2: Υπολογισμός Ιστογράμματος ---
    int px = x_start + tid_x;
    int py = y_start + tid_y;

    // Κάθε thread διαβάζει 1 pixel από Global Memory (Coalesced access λόγω x-continuity)
    if (px < w && py < h) {
        unsigned char val = img[py * w + px];
        // Atomic Add στην Shared Memory: Πολύ πιο γρήγορο από Global Atomic
        // Προσοχή: Εδώ μπορεί να υπάρξουν Bank Conflicts αν πολλά pixels έχουν το ίδιο χρώμα,
        // αλλά είναι αναπόφευκτο στα ιστογράμματα.
        atomicAdd(&s_hist[val], 1);
    }
    __syncthreads(); // Barrier: Περιμένουμε να μετρηθούν όλα τα pixels του tile

    // --- ΒΗΜΑ 3: Clipping & Redistribution ---
    // Χρησιμοποιούμε μόνο 256 threads για αυτή τη διαδικασία
    if (local_tid < 256) {
        int val = s_hist[local_tid];
        int limit = d_CLIP_LIMIT; // Ανάγνωση από Constant Memory

        if (val > limit) {
            s_excess[local_tid] = val - limit; // Κρατάμε το πλεόνασμα
            s_hist[local_tid] = limit;         // "Ψαλιδίζουμε" το ιστόγραμμα
        } else {
            s_excess[local_tid] = 0;
        }
    }
    __syncthreads();

    // --- ΒΗΜΑ 4: Parallel Reduction για το άθροισμα του Excess ---
    // Αντί για loop, χρησιμοποιούμε δέντρο αθροίσματος (O(logN))
    // Το s_excess[local_tid] αθροίζεται μέχρι το s_excess[0] να έχει το σύνολο.
    for (unsigned int stride = 128; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            s_excess[local_tid] += s_excess[local_tid + stride];
        }
        __syncthreads();
    }

    // --- ΒΗΜΑ 5: Αναδιανομή Excess & Δημιουργία CDF/LUT ---
    if (local_tid == 0) {
        // Υπολογισμός μέσης αύξησης
        int total_excess = s_excess[0];
        int avg_inc = total_excess / 256;
        
        // Σειριακός υπολογισμός CDF και LUT (γρήγορο γιατί είναι στην Shared Memory)
        // Δεν κάνουμε parallel scan εδώ για απλότητα κώδικα (μόνο 256 βήματα)
        int cdf = 0;
        int tile_pixels = 0;

        // Υπολογισμός ενεργών pixels (για tiles στα όρια της εικόνας)
        int actual_w = (x_start + TILE_SIZE > w) ? (w - x_start) : TILE_SIZE;
        int actual_h = (y_start + TILE_SIZE > h) ? (h - y_start) : TILE_SIZE;
        tile_pixels = actual_w * actual_h;

        // Δείκτης στο Global Memory LUT για αυτό το Tile
        int* my_lut = &all_luts[(ty * grid_w + tx) * 256];

        for (int i = 0; i < 256; ++i) {
            // Redistribution
            s_hist[i] += avg_inc; 
            
            // CDF Calculation
            cdf += s_hist[i];
            
            // Mapping CDF to 0-255 range
            int val = (int)((float)cdf * 255.0f / tile_pixels + 0.5f);
            if (val > 255) val = 255;
            
            // Εγγραφή στο Global Memory για να το δει το 2ο Kernel
            my_lut[i] = val; 
        }
    }
}

/**
 * KERNEL 2: Διγραμμική Παρεμβολή (Bilinear Interpolation)
 * Στρατηγική: 1 Thread ανά Pixel.
 * Μοτίβο: Gather (κάθε pixel διαβάζει από 4 γειτονικά LUTs).
 */
__global__ void render_clahe_kernel(unsigned char* img_in, unsigned char* img_out, 
                                    int* all_luts, int w, int h, int grid_w, int grid_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    // Μετατροπή pixel coordinates σε grid coordinates
    // Το -0.5 ευθυγραμμίζει τα κέντρα των tiles
    float ty_f = (float)y / TILE_SIZE - 0.5f;
    float tx_f = (float)x / TILE_SIZE - 0.5f;

    int y1 = floorf(ty_f);
    int x1 = floorf(tx_f);
    int y2 = y1 + 1;
    int x2 = x1 + 1;

    // Βάρη παρεμβολής (πόσο κοντά είμαστε στο κάθε tile)
    float y_weight = ty_f - y1;
    float x_weight = tx_f - x1;

    // Clamping: Διαχείριση ορίων (αν είμαστε στην άκρη της εικόνας)
    if (x1 < 0) x1 = 0;
    if (x2 >= grid_w) x2 = grid_w - 1;
    if (y1 < 0) y1 = 0;
    if (y2 >= grid_h) y2 = grid_h - 1;

    // Ανάκτηση αρχικής τιμής
    int val = img_in[y * w + x];

    // Ανάκτηση τιμών από τα 4 γειτονικά LUTs (Global Memory Reads)
    // Τύπος Indexing: (Tile_Y * Grid_Width + Tile_X) * 256 + Intensity
    int tl = all_luts[(y1 * grid_w + x1) * 256 + val]; // Top-Left
    int tr = all_luts[(y1 * grid_w + x2) * 256 + val]; // Top-Right
    int bl = all_luts[(y2 * grid_w + x1) * 256 + val]; // Bottom-Left
    int br = all_luts[(y2 * grid_w + x2) * 256 + val]; // Bottom-Right

    // Υπολογισμός Διγραμμικής Παρεμβολής
    float top = tl * (1.0f - x_weight) + tr * x_weight;
    float bot = bl * (1.0f - x_weight) + br * x_weight;
    float final_val = top * (1.0f - y_weight) + bot * y_weight;

    // Εγγραφή τελικού pixel (Coalesced write)
    img_out[y * w + x] = (unsigned char)(final_val + 0.5f);
}

// Host Function: Καλείται από την main
extern "C" PGM_IMG apply_clahe(PGM_IMG img_in) {
    PGM_IMG img_out;
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img = (unsigned char *)malloc(img_in.w * img_in.h * sizeof(unsigned char));

    // Διαστάσεις πλέγματος (Grid Dimensions)
    int grid_w = (img_in.w + TILE_SIZE - 1) / TILE_SIZE;
    int grid_h = (img_in.h + TILE_SIZE - 1) / TILE_SIZE;

    printf("Grid size: %dx%d tiles\n", grid_w, grid_h);

    // Δέσμευση μνήμης GPU
    unsigned char *d_in, *d_out;
    int *d_luts;
    size_t img_size = img_in.w * img_in.h * sizeof(unsigned char);
    size_t lut_size = grid_w * grid_h * 256 * sizeof(int);

    GPU_Check(cudaMalloc(&d_in, img_size));
    GPU_Check(cudaMalloc(&d_out, img_size));
    GPU_Check(cudaMalloc(&d_luts, lut_size));

    // Αντιγραφή δεδομένων στο Device
    GPU_Check(cudaMemcpy(d_in, img_in.img, img_size, cudaMemcpyHostToDevice));
    
    // Ρύθμιση Constant Memory
    int clip_limit = CLIP_LIMIT;
    GPU_Check(cudaMemcpyToSymbol(d_CLIP_LIMIT, &clip_limit, sizeof(int)));

    // --- Launch Kernel 1: Histogram ---
    // 1 Block per Tile, 32x32 threads per Block
    dim3 hist_block(TILE_SIZE, TILE_SIZE);
    dim3 hist_grid(grid_w, grid_h);
    
    histogram_lut_kernel<<<hist_grid, hist_block>>>(d_in, d_luts, img_in.w, img_in.h, grid_w);
    GPU_Check(cudaPeekAtLastError());

    // --- Launch Kernel 2: Render ---
    // Standard 2D Grid για κάλυψη όλων των pixels
    dim3 render_block(32, 32);
    dim3 render_grid((img_in.w + render_block.x - 1) / render_block.x, 
                     (img_in.h + render_block.y - 1) / render_block.y);

    render_clahe_kernel<<<render_grid, render_block>>>(d_in, d_out, d_luts, img_in.w, img_in.h, grid_w, grid_h);
    GPU_Check(cudaPeekAtLastError());
    
    // Συγχρονισμός και επιστροφή
    GPU_Check(cudaDeviceSynchronize());
    GPU_Check(cudaMemcpy(img_out.img, d_out, img_size, cudaMemcpyDeviceToHost));

    // Απελευθέρωση μνήμης GPU
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_luts);

    return img_out;
}
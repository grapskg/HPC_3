#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "clahe.h"
#include "gputimer.h"

double get_time_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]){
    if (argc != 3) {
        printf("Usage: %s <input.pgm> <output.pgm>\n", argv[0]);
        return 1;
    }

    //load image
    printf("Loading image...\n");
    PGM_IMG img_in = read_pgm(argv[1]);
    
    // ============================================
    // 1. EXEC CPU (Serial)
    // ============================================
    printf("\n--- Starting CPU Execution ---\n");
    double start_cpu = get_time_sec();
    
    PGM_IMG img_out_cpu = apply_clahe_cpu(img_in);
    
    double time_cpu = get_time_sec() - start_cpu;
    printf("CPU Time:       %.6f sec\n", time_cpu);
    printf("CPU Throughput: %.2f MPixels/s\n", (img_in.w * img_in.h) / (time_cpu * 1e6));
    
    // ============================================
    // 2. EXEC GPU (Parallel)
    // ============================================
    
    int grid_w = (img_in.w + TILE_SIZE - 1) / TILE_SIZE;
    int grid_h = (img_in.h + TILE_SIZE - 1) / TILE_SIZE;
    printf("Image: %dx%d, Grid: %dx%d\n", img_in.w, img_in.h, grid_w, grid_h);

    printf("\n--- Starting GPU Execution ---\n");
    
    // PREPARE GPU
    unsigned char *d_in, *d_out;
    int *d_luts;
    size_t img_size = img_in.w * img_in.h * sizeof(unsigned char);
    size_t lut_size = grid_w * grid_h * 256 * sizeof(int);
    // START GPU TIMER
    {
    GpuTimer timer;
    timer.Start();
    
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);
    cudaMalloc(&d_luts, lut_size);

    cudaMemcpy(d_in, img_in.img, img_size, cudaMemcpyHostToDevice);

    dim3 hist_block(TILE_SIZE, TILE_SIZE);
    dim3 hist_grid(grid_w, grid_h);
    
    histogram_lut_kernel<<<hist_grid, hist_block>>>(d_in, d_luts, img_in.w, img_in.h, grid_w, CLIP_LIMIT);

    dim3 render_block(32, 32);
    dim3 render_grid((img_in.w + 31) / 32, (img_in.h + 31) / 32);
    render_clahe_kernel<<<render_grid, render_block>>>(d_in, d_out, d_luts, img_in.w, img_in.h, grid_w, grid_h);

    cudaDeviceSynchronize();
    
    //WRITE BACK OUTPUT
    PGM_IMG img_out_gpu;
    img_out_gpu.w = img_in.w; img_out_gpu.h = img_in.h;
    img_out_gpu.img = (unsigned char *)malloc(img_size);
    cudaMemcpy(img_out_gpu.img, d_out, img_size, cudaMemcpyDeviceToHost);
    timer.Stop();
    float time_gpu = timer.Elapsed();
    printf("GPU Time:       %.6f ms\n", time_gpu);
    time_gpu = time_gpu / 1000;//turn to sec
    printf("GPU Throughput: %.2f MPixels/s\n", (img_in.w * img_in.h) / (time_gpu * 1e6));
   
    // ============================================
    // 3. COMPARE
    // ============================================
    printf("\n--- Results ---\n");
    printf("Speedup: %.2fx faster on GPU\n", time_cpu / time_gpu);

    // WRITE BACK GPU 
    write_pgm(img_out_gpu, argv[2]);
    printf("Output image saved to %s\n", argv[2]);

    // Cleanup
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_luts);
    free_pgm(img_in);
    free_pgm(img_out_cpu);
    free_pgm(img_out_gpu);
    }
    return 0;
}
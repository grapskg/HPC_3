#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include "clahe.h"

// Helper function for timing
double get_time_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Helper: Read PGM
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    PGM_IMG result;
    int v_max;

    in_file = fopen(path, "rb");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip P5*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d",&v_max);
    fgetc(in_file); 

    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    fread(result.img, sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

// Helper: Write PGM
void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n", img.w, img.h);
    fwrite(img.img, sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

// Helper: Free PGM Memory
void free_pgm(PGM_IMG img) {
    if(img.img) free(img.img);
}

int main(int argc, char *argv[]){
    PGM_IMG img_in, img_out;
    double start, end, elapsed;

    if (argc != 3) {
        printf("Usage: %s <input.pgm> <output.pgm>\n", argv[0]);
        return 1;
    }

    printf("Loading image...\n");
    img_in = read_pgm(argv[1]);
    
    printf("Running GPU CLAHE...\n");
    start = get_time_sec();
    
    // Αυτή η συνάρτηση καλείται τώρα από το clahe.cu
    img_out = apply_clahe(img_in);
    
    end = get_time_sec();
    elapsed = end - start;
    
    printf("Processing time: %.6f seconds\n", elapsed);
    printf("Throughput: %.2f MPixels/s\n", (img_in.w * img_in.h) / (elapsed * 1e6));

    write_pgm(img_out, argv[2]);
    printf("Result saved to %s\n", argv[2]);

    free_pgm(img_in);
    free_pgm(img_out);

    return 0;
}
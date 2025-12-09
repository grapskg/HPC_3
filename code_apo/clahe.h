#ifndef CLAHE_H
#define CLAHE_H

#include <stdio.h>
#include <stdlib.h>

// Μέγεθος Tile: 32x32 pixels (Ταιριάζει τέλεια με το warp size και max threads/block)
#define TILE_SIZE 32    
// Όριο αντίθεσης (Contrast Limit)
#define CLIP_LIMIT 4 // Μπορείς να το πειράξεις (συνήθως 2-4)

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;

// I/O functions
PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

// Χρήση extern "C" για να επιτρέψουμε τη σύνδεση μεταξύ C (main) και CUDA (clahe.cu)
#ifdef __cplusplus
extern "C" {
#endif

    PGM_IMG apply_clahe(PGM_IMG img_in);

#ifdef __cplusplus
}
#endif

#endif
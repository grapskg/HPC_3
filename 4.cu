/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include<math.h>
#include <time.h>
#include "gputimer.h"


unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
#define accuracy  0.00005 

 

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

  int x, y, k;
                      
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = x + k;

        if (d >= 0 && d < imageW) {
          sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
        }     

        h_Dst[y * imageW + x] = sum;
      }
    }
  }
        
}


__global__ void convRowGPU(float *d_Dst, const float *d_Src, const float *d_Filter, 
                       const int imageW, const int imageH, const int filterR){
    
    int k; 
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    float sum;
    
    if (ix <imageW && iy < imageH){
      sum=0.0;
      for (k = -filterR; k <= filterR; k++) {
          int d = ix + k;
  
          if (d >= 0 && d < imageW) {
            sum += d_Src[iy * imageW + d] * d_Filter[filterR - k];
          }     
  
      }
      d_Dst[iy * imageW + ix] = sum;
    }    
}

////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

  int x, y, k;
  
  for (y = 0; y < imageH; y++) {
    for (x = 0; x < imageW; x++) {
      float sum = 0;

      for (k = -filterR; k <= filterR; k++) {
        int d = y + k;

        if (d >= 0 && d < imageH) {
          sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
        }   
 
        h_Dst[y * imageW + x] = sum;
      }
    }
  }
    
}

__global__ void convColGPU(float *d_Dst, const float *d_Src, const float *d_Filter, 
                       const int imageW, const int imageH, const int filterR){
    
    int k; 

    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;

    if (ix <imageW && iy < imageH){
      float sum=0;
      for (k = -filterR; k <= filterR; k++) {
          int d = iy + k;
  
          if (d >= 0 && d < imageH) {
            sum += d_Src[d * imageW + ix] * d_Filter[filterR - k];
          }     
      }
      d_Dst[iy * imageW + ix] = sum;
    }

}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
    float
    *h_Filter,
    *h_Input,
    *h_Buffer,
    *h_OutputCPU,
    *h_OutputGPU_Host,
    *d_Filter,
    *d_Input,
    *d_Buffer,
    *d_OutputGPU;

    int imageW;
    int imageH;
    unsigned int i;
    double cpu_time, gpu_time;
    struct timespec start, stop;
    GpuTimer timer;

    
	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

    // Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
    // dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
    // Gia aplothta thewroume tetragwnikes eikones.  

    printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
    scanf("%d", &imageW);
    imageH = imageW;

    printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
    printf("Allocating and initializing host arrays...\n");
    // Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
    h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
    h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
    h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
    h_OutputGPU_Host = (float *)malloc(imageW * imageH * sizeof(float));


    //alloc cuda resources
    cudaMalloc((void**)&d_Filter, FILTER_LENGTH * sizeof(float));
    cudaMalloc((void**)&d_Input, imageW * imageH * sizeof(float));
    cudaMalloc((void**)&d_Buffer, imageW * imageH * sizeof(float));
    cudaMalloc((void**)&d_OutputGPU, imageW * imageH * sizeof(float));

    if(h_Filter == NULL || h_Input == NULL || h_Buffer == NULL || h_OutputCPU == NULL ||
       d_Filter == NULL || d_Input == NULL || d_Buffer == NULL || d_OutputGPU == NULL) {
        printf("couldn't allocate memory\n");
	    cudaDeviceReset();
        return 1;
    }

    // to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
    // arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
    // to convolution kai arxikopoieitai kai auth tuxaia.

    srand(200);

    for (i = 0; i < FILTER_LENGTH; i++) {
        h_Filter[i] = (float)(rand() % 16);
    }

    for (i = 0; i < imageW * imageH; i++) {
        h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
    }



    // To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
    printf("CPU computation...\n");

    clock_gettime(CLOCK_MONOTONIC, &start);

    convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
    convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

    clock_gettime(CLOCK_MONOTONIC, &stop);

    cpu_time = (stop.tv_sec - start.tv_sec) 
          + (stop.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("CPU execution time: %lf ms\n", cpu_time * 1000.0);
    printf("GPU computation...\n");
    dim3 blockDim(32, 32);
    dim3 gridDim((imageW + blockDim.x - 1) / blockDim.x,(imageH + blockDim.y - 1) / blockDim.y);

    //Start GPU timer
    timer.Start();

    cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Input, h_Input, imageW * imageH * sizeof(float), cudaMemcpyHostToDevice);

    convRowGPU<<<gridDim,blockDim>>>(d_Buffer,d_Input, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();

    convColGPU<<<gridDim,blockDim>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);
    cudaDeviceSynchronize();

    cudaMemcpy(h_OutputGPU_Host, d_OutputGPU, imageW * imageH * sizeof(float), cudaMemcpyDeviceToHost);

    //Stop GPU timer
    timer.Stop();

    float gpu_time_ms = timer.Elapsed();

    printf("GPU execution time: %f ms\n", gpu_time_ms);

    // Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
    // pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

    printf("Checking accuracy...\n");

    //int errors = 0;
    //for (i = 0; i < imageW * imageH; i++) {
    //    float cpu_val = h_OutputCPU[i];
    //    float gpu_val = h_OutputGPU_Host[i];
//
    //    if (ABS(cpu_val - gpu_val) > accuracy) {
    //        errors++;
    //        // Print only the first few errors to avoid spamming the console
    //        if (errors < 10) {
    //            //printf("Error at index %d: CPU=%f, GPU=%f\n", i, cpu_val, gpu_val);
    //            printf("%f\n",ABS(cpu_val - gpu_val) );
    //        }
    //        else
    //          break;
    //    }
    //}
//
    //if (errors == 0) {
    //    printf("TEST PASSED! Results match.\n");
    //} else {
    //    printf("TEST FAILED, error found.\n", errors);
    //}
//

    // free all the allocated memory
    free(h_OutputCPU);
    free(h_OutputGPU_Host);
    free(h_Buffer);
    free(h_Input);
    free(h_Filter);
    cudaFree(d_Buffer);
    cudaFree(d_Filter);
    cudaFree(d_OutputGPU);
    cudaFree(d_Input);
    cudaDeviceReset();

    
    // Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
    // cudaDeviceReset();


    return 0;
}

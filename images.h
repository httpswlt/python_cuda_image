#include<stdio.h>
#include <stdlib.h>
typedef struct Image{
    float* data;                    // output data
    int srcWidth;                   // width of original image
    int srcHeight;                  // height of original image
    int dstWidth;                   // width of target image
    int dstHeight;                  // height of target image
    float* cu_dst;                  // output image by cuda
    unsigned char* cu_src;          // accept original by cuda
    unsigned char* cu_dst_resize;   // save the resize image by cuda
}Image;

typedef struct{
    int prob;
    int classify;
    float x, y, w, h,conf;
} box;

typedef struct{
    int nums;
    int cols;
    float* result;
} BBOX;

Image* init_cuda_memory(int srcWidth,int srcHeight,int dstWidth,int dstHeight);
//Image* resize(const unsigned char*src,int srcWidth,int srcHeight,int dstWidth,int dstHeight);
Image* resize_cu(const unsigned char*src,Image* img);

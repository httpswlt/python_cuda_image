#include "images.h"

Image* init_cuda_memory(int srcWidth,int srcHeight,int dstWidth,int dstHeight){

    Image* im = (Image*)calloc(1,sizeof(Image));
    im->data = (float*)calloc(1,dstWidth*dstHeight*3*sizeof(float));
    im->dstHeight = dstHeight;
    im->dstWidth = dstWidth;
    im->srcWidth = srcWidth;
    im->srcHeight = srcHeight;
    cudaMalloc((void**)&(im->cu_src),srcWidth*srcHeight*3*sizeof(unsigned char));
//    cudaMemset(im->cu_src,0,srcWidth*srcHeight*3*sizeof(unsigned char));
    cudaMalloc((void**)&(im->cu_dst_resize),dstWidth*dstHeight*3*sizeof(unsigned char));
//    cudaMemset(im->cu_dst_resize,0,dstWidth*dstHeight*3*sizeof(unsigned char));
    cudaMalloc((void**)&(im->cu_dst),dstWidth*dstHeight*3*sizeof(float));
//    cudaMemset(im->cu_dst,0,dstWidth*dstHeight*3*sizeof(unsigned char));
    return im;
}


float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1;
    float l2 = x2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1;
    float r2 = x2 + w2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

int nms_comparator(const void *pa, const void *pb)
{
    box a = *(box *)pa;
    box b = *(box *)pb;
    float diff = a.conf - b.conf;
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}

void do_nms_sort(float* dets, BBOX* bbox,int classes, float nms){
    int i,j,k;
    box det_arr[bbox->nums];
    for (i = 0; i < bbox->nums; i++){
            box a;
            a.x = dets[i*bbox->cols];
            a.y = dets[i*bbox->cols + 1];
            a.w = dets[i*bbox->cols + 2];
            a.h = dets[i*bbox->cols + 3];
            a.conf = dets[i*bbox->cols + 4];
            a.classify = dets[i*bbox->cols + 5];
            a.prob = 1;
            det_arr[i] = a;
    }
    for(k = 0; k < classes; ++k){
        qsort(det_arr,bbox->nums,sizeof(box),nms_comparator);
        for(i = 0; i < bbox->nums; ++i){
            if(det_arr[i].prob == 0) continue;
            box a = det_arr[i];
            for(j = i+1; j < bbox->nums; ++j){
                box b = det_arr[j];
                if (box_iou(a, b) > nms){
                    det_arr[j].prob = 0;
                }
            }
        }
    }
    int count = 0;
    j = 0;
    for (i = 0; i < bbox->nums; ++i){
        if (det_arr[i].prob != 0){
            count++;
            k = j * bbox->cols;
            *(bbox->result + k) = det_arr[i].x;
            *(bbox->result + k+1) = det_arr[i].y;
            *(bbox->result + k+2) = det_arr[i].w;
            *(bbox->result + k+3) = det_arr[i].h;
            *(bbox->result + k+4) = det_arr[i].conf;
            *(bbox->result + k+5) = det_arr[i].classify;
        }
        else{
            j --;
        }
        j ++;

    }
    bbox->nums = count;
}


























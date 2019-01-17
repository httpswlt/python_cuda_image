extern "C" {
    #include "images.h"
}

__global__ void resizeGPU(const unsigned char*src,int srcWidth,int srcHeight,
                          unsigned char *dst_resize,int dstWidth,int dstHeight,
                          float w_ratio,float h_ratio,float *dst)
{
	const int x = blockIdx.x*blockDim.x+threadIdx.x;
	const int y = blockIdx.y*blockDim.y+threadIdx.y;
	if(x < dstWidth && y < dstHeight){
        float srcXf=  x * w_ratio;
        float srcYf =  y * h_ratio;
        int srcX = (int)srcXf;
        int srcY = (int)srcYf;
        float u= srcXf - srcX;
        float v = srcYf - srcY;
        int dstOffset = (y*dstWidth + x);
        int offset;
        //resize(inter_linear)
        for(int i = 0;i < 3;++i){
            offset = dstOffset*3 + i;
            dst_resize[offset] = (1-u)*(1-v)*src[(srcY*srcWidth+srcX)*3 + i];
            dst_resize[offset] += (1-u)*v*src[((srcY+1)*srcWidth+srcX)*3 + i];
            dst_resize[offset] += u*(1-v)*src[(srcY*srcWidth+srcX+1)*3 + i];
            dst_resize[offset] += u*v*src[((srcY+1)*srcWidth+srcX+1)*3 + i];
        }

        // BGR_RGB and channel split.
        int image_patch = dstWidth*dstHeight;
        offset = dstOffset;
//        BGR2RGB
//        dst[offset*3 + 2] = dst_resize[offset*3];
//        dst[offset*3 + 1] = dst_resize[offset*3+1];
//        dst[offset*3] = dst_resize[offset*3+2];
        //merge

        dst[offset] = dst_resize[offset*3+2] / 255.0f;
        dst[offset + image_patch] = dst_resize[offset*3+1] / 255.0f;
        dst[offset + image_patch*2] = dst_resize[offset*3]  / 255.0f;

	}
}

Image* resize_cu(const unsigned char*src,Image* img){

    cudaMemcpy(img->cu_src,src,img->srcWidth*img->srcHeight*3*sizeof(unsigned char),cudaMemcpyHostToDevice);
    int uint = 16;
    dim3 grid((img->dstWidth+uint-1)/uint,(img->dstHeight+uint-1)/uint);
    dim3 block(uint,uint);
    float w_ratio = (float)img->srcWidth/img->dstWidth;
    float h_ratio = (float)img->srcHeight/img->dstHeight;

    resizeGPU<<<grid,block>>>(img->cu_src, img->srcWidth,img->srcHeight,
    img->cu_dst_resize,img->dstWidth,img->dstHeight,w_ratio,h_ratio,img->cu_dst);
    cudaMemcpy(img->data,img->cu_dst,img->dstHeight*img->dstWidth*3*sizeof(float),cudaMemcpyDeviceToHost);
    return img;

}
















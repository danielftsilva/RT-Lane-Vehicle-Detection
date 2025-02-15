#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

//Sampler
__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
    CLK_ADDRESS_CLAMP_TO_EDGE | //Clamp to zeros
    CLK_FILTER_NEAREST;

//Hough Transform (basic algorithm) for BGRA (uchar4) images
//Returns: accumulator

__kernel void hough_GPU(
    __read_only image2d_t image,                    //binary BGRA image (R,G,B = 255 or 0)
    int w,                                          //image width
    int h,                                          //image height
    int filter_hLow,
    int filter_hHigh,
    __global int* accumulator)                     //accumulator
{
//Get coordinates:
int iX = get_global_id(0);      //column
int iY = get_global_id(1);      //row

// Limitar a imagem!!

//If pixel in image
if(iX<w && iY<h && iX>0 && iY>0 && iY>filter_hHigh && iY<(h-filter_hLow)){
    uint4 pixel = read_imageui( image, sampler, (int2)(iX,iY));

    //If pixel is white
    if(pixel.x == 255 && pixel.y == 255 && pixel.z == 255){
        double theta;
        int r;

        //Iterate through every angle (avoid 90º)
        for(int i = 0; i<85; i++){
            theta = i*(3.14159265359/180);              //theta in rads
            r = iX*cos(theta) + iY*sin(theta);          //rho for this theta
            atomic_add(&accumulator[(r*180 + i)], 1);

        }

        /*  RIGHT LINE
        for(int i = 95; i<180; i++){
           theta = i*(3.14159265359/180);
            r = iX*cos(theta) + iY*sin(theta);
            atomic_add(&accumulator[(r*180 + i)], 1);
        }*/

    }
}
}
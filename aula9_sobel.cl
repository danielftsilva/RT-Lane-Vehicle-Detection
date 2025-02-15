//Sampler
__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE | //Natural coordinates
    CLK_ADDRESS_CLAMP_TO_EDGE | //Clamp to zeros
    CLK_FILTER_NEAREST;

//Sobel for BGRA (uchar4) images

__kernel void sobel_BGRA(
    __read_only image2d_t image,
    __write_only image2d_t imageOut,
    int w,
    int h,
    int t1,
    int t2)
{
//Get coordinates
int iX = get_global_id(0);      //column
int iY = get_global_id(1);      //row


if(iX<w && iY<h){
    int x = iX;
    int y = iY;

    uint4 Pixel00 = read_imageui(image, sampler, (int2)(x - 1, y - 1));
    uint4 Pixel01 = read_imageui(image, sampler, (int2)(x, y - 1));
	uint4 Pixel02 = read_imageui(image, sampler, (int2)(x + 1, y - 1));

	uint4 Pixel10 = read_imageui(image, sampler, (int2)(x - 1, y));
	uint4 Pixel12 = read_imageui(image, sampler, (int2)(x + 1, y));

	uint4 Pixel20 = read_imageui(image, sampler, (int2)(x - 1, y + 1));
	uint4 Pixel21 = read_imageui(image, sampler, (int2)(x, y + 1));
	uint4 Pixel22 = read_imageui(image, sampler, (int2)(x + 1, y + 1));

    uint4 Gx = Pixel00 + (2 * Pixel10) + Pixel20 - Pixel02 - (2 * Pixel12) - Pixel22;
    uint4 Gy = Pixel00 + (2 * Pixel01) + Pixel02 - Pixel20 - (2 * Pixel21) - Pixel22;

    uint4 G = (uint4)(0, 0, 0, Pixel00.w);
    G.x = sqrt((float)(Gx.x * Gx.x + Gy.x * Gy.x)); // B
    G.y = sqrt((float)(Gx.y * Gx.y + Gy.y * Gy.y)); // G
	G.z = sqrt((float)(Gx.z * Gx.z + Gy.z * Gy.z)); // R

	double diff = abs((int)(G.z-G.y)) + abs((int)(G.z-G.x)) + abs((int)(G.y-G.x));
    double average = (G.x+G.y+G.z)/3;

    if(diff>t1 && average>t2){
       write_imageui( imageOut, (int2)(iX,iY) , (uint4)(255,255,255,0));
    }
    else
        write_imageui( imageOut, (int2)(iX,iY) , (uint4)(0,0,0,0));

}
}
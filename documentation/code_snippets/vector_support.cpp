using namespace hipacc;
using namespace hipacc::math;

ushort4 pixel_s = { 0, 0, 0, 0};

uchar4 pixel;
pixel.x = 204;
pixel.y = 0;
pixel.z = 0;
pixel.w = 0;

float4 tmp;
// using sin from hipacc::math
tmp = sin(convert_float4(pixel));
// calling sin from hipacc::math directly
tmp = hipacc::math::sin(convert_float4(pixel));

pixel_s = convert_uchar(tmp);

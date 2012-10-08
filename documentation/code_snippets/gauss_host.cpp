const int width=1024, height=1024, size_x=3, size_y=3;

// pointers to raw image data
float *host_in = ...;
float *host_out = ...;
// pointer to Gaussian filter mask
float *filter_mask = ...;

// input and output images
Image<float> IN(width, height);
Image<float> OUT(width, height);

// initialize input image
IN = host_in; // operator=

// filter Mask for Gaussian filter
Mask<float> GMask(size_x, size_y);
GMask = filter_mask;

// Boundary handling mode for out of bounds accesses
BoundaryCondition<float> BcInMirror(IN, GMask, BOUNDARY_MIRROR);

// define region of interest
IterationSpace<float> IsOut(OUT);

// Accessor used to access image pixels with the defined boundary handling mode
Accessor<float> AccIn(BcInMirror);

// define kernel
GaussianFilter GF(IsOut, AccIn, GMask, size_x, size_y);

// execute kernel
GF.execute();

// retrieve output image
host_out = OUT.getData();


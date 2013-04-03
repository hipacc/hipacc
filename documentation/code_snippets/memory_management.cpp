// input Image
int width, height;
uchar *image = read_image(&width, &height, "input.pgm");
Image<uchar> IN(width, height);

// copy data to the device: host -> device
IN = image;

// define second Image
Image<uchar> TMP(width, height);

// copy from IN to TMP: device -> device
TMP = IN;


// define ROI on IN (Accessor)
Accessor<uchar> AccIn(IN, roi_width, roi_height, offset_x, offset_y);

// define ROI on TMP (Accessor)
Accessor<uchar> AccTmp(TMP, roi_width, roi_height, 0, 0);

// copy from ROI on IN to ROI on TMP: device -> device
AccTmp = AccIn;


// output image
Image<uchar> OUT(roi_width, roi_height);

// copy from ROI on TMP to OUT: device -> device
OUT = AccTmp;

// copy from Accessor to Image: device -> device
AccTmp = OUT;

// copy data from device to host: device -> host
OUT.getData();

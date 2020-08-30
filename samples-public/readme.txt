The OpenCV library is required for the samples to produce visual output.

---

On Ubuntu

Make sure that "libopencv-dev" is installed:
    sudo apt install libopencv-dev

---

On Windows

Download and install latest release from OpenCV's website (https://opencv.org).
After unpacking to <INSTDIR>, make sure to tell CMake where to find it:
    cmake <SRCDIR> -DOpenCV_DIR="<INSTDIR>\opencv\build"

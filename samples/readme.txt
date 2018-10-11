The OpenCV library is required for the samples to produce visual output.

---

On Ubuntu

Make sure that "libopencv-dev" is installed:
    sudo apt install libopencv-dev

---

On Windows

Download and install latest release from OpenCV's website (https://opencv.org).
After unpacking to <INSTDIR>, set the environment variable "OPENCV_DIR":
    setx -m OPENCV_DIR <INSTDIR>\opencv\build\x64\vc14

Make sure that "%OPENCV_DIR%\bin" is added to the system's "PATH" variable.


#ifndef HIPACC_HELPER_HPP
#define HIPACC_HELPER_HPP

#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <vector>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif


// get time in milliseconds
double time_ms () {
    auto time = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(time).count();
}


template<typename T>
T* load_data(const unsigned int width, const unsigned int height,
             const unsigned int chan=1, const std::string filename="") {
    if (std::is_same<T,unsigned char>::value) {
        if (chan != 1 && chan != 4) {
            std::cerr << "Only grayscale or BRGA supported for uchar" << std::endl;
            exit(-1);
        }
    } else if (std::is_same<T,float>::value) {
        if (chan != 1) {
            std::cerr << "Only grayscale supported for float" << std::endl;
            exit(-1);
        }
    } else {
        std::cerr << "Type not supported" << std::endl;
        exit(-1);
    }

    T* data = new T[width*height*chan];

#ifdef USE_OPENCV
    if (!filename.empty()) {
        cv::Mat image = cv::imread(filename.c_str(), 1);

        if (!image.data) {
            std::cerr << "No image data" << std::endl;
            exit(-1);
        }

        if (image.cols != width || image.rows != height) {
            std::cerr << "Image size mismatch" << std::endl;
            exit(-1);
        }

        if (image.channels() == chan) {
            // copy entire array
            std::memcpy(data, (T*)image.data, width*height*chan*sizeof(T));
        } else {
            if (chan == 1) {
                // convert to grayscale
                cv::Mat gray_image;
                cv::cvtColor(image, gray_image, CV_BGR2GRAY);
                for (unsigned int p = 0; p < width*height; ++p) {
                    data[p] = (T)gray_image.data[p];
                }
            } else {
                // match BGR to BRGA
                unsigned char* d = (unsigned char*)data;
                for (unsigned int p = 0; p < width*height; ++p) {
                    for (unsigned int c = 0; c < chan; ++c) {
                        d[p*chan+c] = c < image.channels() ? image.data[p*image.channels()+c] : 0;
                    }
                }
            }
        }
      return data;
    }
#endif

    // random data, channels will be ignored
    for (unsigned int p = 0; p < width * height; ++p) {
        if (std::is_same<T,float>::value) {
            data[p] = ((T)std::rand())/RAND_MAX;
        } else {
            data[p] = (std::rand())%256;
        }
    }

    return data;
}


template<typename T>
void save_data(const unsigned int width, const unsigned int height,
               const unsigned int chan, T* data, const std::string filename) {
#ifdef USE_OPENCV
    cv::Mat image;

    if (std::is_same<T,unsigned char>::value) {
        switch (chan) {
            case 1:
                image.create(height, width, CV_8UC1);
                break;
            case 4:
                image.create(height, width, CV_8UC4);
                break;
            default:
                std::cerr << "Only grayscale or RGBA supported for uchar" << std::endl;
                exit(-1);
                break;
        }
    } else if (std::is_same<T,float>::value) {
        if (chan != 1) {
            std::cerr << "Only grayscale supported for float" << std::endl;
            exit(-1);
        }
        image.create(height, width, CV_32FC1);
    } else {
        std::cerr << "Type not supported" << std::endl;
        exit(-1);
    }

    std::memcpy((T*)image.data, data, width*height*chan*sizeof(T));

    cv::imwrite(filename.c_str(), image);
#endif
}


template<typename T>
bool show_data(const unsigned int width, const unsigned int height,
               const unsigned int chan, T* data, const std::string title,
               const int wait=0) {
#ifdef USE_OPENCV
    cv::Mat image;

    if (std::is_same<T,unsigned char>::value) {
        switch (chan) {
            case 1:
                image.create(height, width, CV_8UC1);
                break;
            case 4:
                image.create(height, width, CV_8UC4);
                break;
            default:
                std::cerr << "Only grayscale or RGBA supported for uchar" << std::endl;
                exit(-1);
                break;
        }
    } else if (std::is_same<T,float>::value) {
        if (chan != 1) {
            std::cerr << "Only grayscale supported for float" << std::endl;
            exit(-1);
        }
        image.create(height, width, CV_32FC1);
    } else {
        std::cerr << "Type not supported" << std::endl;
        exit(-1);
    }

    std::memcpy((T*)image.data, data, width*height*chan*sizeof(T));

    if (height > 900) {
        float scale = 900/(float)height;
        cv::namedWindow(title.c_str(), cv::WINDOW_NORMAL);
        cv::resizeWindow(title.c_str(), width*scale, height*scale);
    }

    cv::imshow(title.c_str(), image);

    return cv::waitKey(wait) >= 0;
#else
    return true;
#endif
}


template<typename T>
void compare_results(const T* output, const T* reference,
                     const unsigned int width=1, const unsigned int height=1,
                     const unsigned int border_x=0,
                     const unsigned int border_y=0) {
    std::cout << "Comparing results ..." << std::endl;
    for (unsigned int y = border_y; y < height-border_y; ++y) {
        for (unsigned int x = border_x; x < width-border_x; ++x) {
            bool failed = true;

            if (std::is_same<T,float>::value) {
                float ref = reference[y*width + x];
                failed = abs(ref-output[y*width + x]) > (0.001f*ref);
            } else {
                int diff = (int)reference[y*width + x]-(int)output[y*width + x];
                failed = abs((float)diff) > 1.0f;
            }

            if (failed) {
                std::cerr << "Test FAILED, at (" << x << "," << y << "): "
                          << (float)reference[y*width + x] << " vs. "
                          << (float)output[y*width + x] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
    std::cout << "Test PASSED" << std::endl;
}


#endif // HIPACC_HELPER_HPP

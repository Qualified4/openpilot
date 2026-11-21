// cv::remap (float maps, INTER_LINEAR, BORDER_REPLICATE) for src/native.rs — OpenCV 4.13.0 core + imgproc built with
// the KleidiCV 0.7.0 HAL, the call cv2.remap makes in the device's opencv-python-headless 4.13.0.
#include <cstddef>
#include <cstdint>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

extern "C" void stripnet_opencv_set_threads(int n) { cv::setNumThreads(n); }

extern "C" int stripnet_opencv_remap(const uint8_t *src, size_t stride, size_t w, size_t h, uint8_t *dst, size_t dst_w,
                                     size_t dst_h, size_t channels, const float *mx, const float *my) {
  try {
    const int type = CV_8UC(static_cast<int>(channels));
    cv::Mat s(static_cast<int>(h), static_cast<int>(w), type, const_cast<uint8_t *>(src), stride);
    cv::Mat d(static_cast<int>(dst_h), static_cast<int>(dst_w), type, dst);
    cv::Mat x(static_cast<int>(dst_h), static_cast<int>(dst_w), CV_32F, const_cast<float *>(mx));
    cv::Mat y(static_cast<int>(dst_h), static_cast<int>(dst_w), CV_32F, const_cast<float *>(my));
    cv::remap(s, d, x, y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return d.data == dst ? 0 : 2;  // a reallocated dst would mean the result went elsewhere
  } catch (...) {
    return 1;
  }
}

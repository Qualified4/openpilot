// kleidicv_remap_f32_u8 (INTER_LINEAR, BORDER_REPLICATE) for src/native.rs — KleidiCV 0.7.0, linked statically.
#include <cstddef>
#include <cstdint>
#include "kleidicv/kleidicv.h"

extern "C" int stripnet_kleidicv_remap(const uint8_t *src, size_t stride, size_t w, size_t h, uint8_t *dst, size_t dst_w,
                                       size_t dst_h, size_t channels, const float *mx, const float *my) {
  static const uint8_t border[4] = {0, 0, 0, 0};
  kleidicv_error_t e = kleidicv_remap_f32_u8(src, stride, w, h, dst, dst_w * channels, dst_w, dst_h, channels, mx,
                                             dst_w * sizeof(float), my, dst_w * sizeof(float),
                                             KLEIDICV_INTERPOLATION_LINEAR, KLEIDICV_BORDER_TYPE_REPLICATE, border);
  return e == KLEIDICV_OK ? 0 : static_cast<int>(e) + 1000;
}

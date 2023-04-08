#include <utils.h>

namespace fft {
namespace utils {
unsigned int bit_reverse(unsigned int num, unsigned int digits) {
  unsigned int ret = 0;

  for (unsigned int i = 0; i < digits; ++i) {
    ret |= ((num & (1 << i)) > 0) * (1 << (digits - 1 - i));
  }

  return ret;
}
} // namespace utils
} // namespace fft

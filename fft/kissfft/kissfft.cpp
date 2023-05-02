#include "kissfft.hpp"
#include "kissfft_first.hpp"

#include <config.hpp>
#include <utils.h>

namespace fft {
namespace alg {
namespace cpu {
void kissfftc::execute(std::vector<std::complex<conf::DT>> &data) {
  kissfft<conf::DT, utils::int_<conf::size>>::execute(data);
};
} // namespace cpu
} // namespace alg
} // namespace fft

#include "fftw.hpp"
#include "fftw_first.hpp"

#include <config.hpp>
#include <utils.h>

namespace fft {
namespace alg {
namespace cpu {
void fftwc::execute(std::vector<std::complex<conf::DT>> &data) {
  fftw<conf::DT, utils::int_<conf::size>>::execute(data);
};
} // namespace cpu
} // namespace alg
} // namespace fft

#pragma once

#include <fft.hpp>
#include <kiss_fft.h>
#include <utils.h>

namespace fft {
namespace alg {
namespace cpu {

template <typename DT, typename Size>
struct kissfft : fft_functor<kissfft<DT, Size>,
                             fft_properties<DT, Size::value, false, false>> {
  static void exec_impl(std::vector<std::complex<DT>> &data) {
    constexpr auto size = Size::value;
    kiss_fft_cfg cfg = kiss_fft_alloc(size, 0, NULL, NULL);

    auto cx_in = (kiss_fft_cpx *)malloc(sizeof(kiss_fft_cpx) * size);

    for (int i = 0; i < size; ++i) {
      cx_in[i].r = data[i].real();
      cx_in[i].i = data[i].imag();
    }

    kiss_fft(cfg, cx_in, cx_in);

    for (int i = 0; i < size; ++i) {
      data[i].real(cx_in[i].r);
      data[i].imag(cx_in[i].i);
    }
  }
};
} // namespace cpu
} // namespace alg
} // namespace fft

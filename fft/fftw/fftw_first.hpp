#pragma once

#include <fft.hpp>
#include <fftw3.h>
#include <utils.h>

namespace fft {
namespace alg {
namespace cpu {

template <typename DT, typename Size>
struct fftw : fft_functor<fftw<DT, Size>,
                          fft_properties<DT, Size::value, false, false>> {
  static void exec_impl(std::vector<std::complex<DT>> &data) {
    constexpr auto size = Size::value;

    fftw_complex *in, *out;
    fftw_plan p;

    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * size);

    for (int i = 0; i < size; ++i) {
      in[i][0] = data[i].real();
      in[i][1] = data[i].imag();
    }
    p = fftw_plan_dft_1d(size, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p);

    for (int i = 0; i < size; ++i) {
      data[i] = {out[i][0], out[i][1]};
    }

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
  }
};
} // namespace cpu
} // namespace alg
} // namespace fft

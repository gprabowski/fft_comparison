#pragma once

#include "cpu_fft.h"

#include <string_view>

namespace fft {

template <typename> struct fft_trait_name;

template <> struct fft_trait_name<fft::alg::cpu::ibbc_first> {
  static constexpr std::string_view value = "DIT NR from Inside FFT";
};

template <> struct fft_trait_name<fft::alg::cpu::ibbc_second> {
  static constexpr std::string_view value = "DIT RN from Inside FFT";
};

template <> struct fft_trait_name<fft::alg::cpu::ibbc_third> {
  static constexpr std::string_view value = "DIT NN from Inside FFT";
};

template <> struct fft_trait_name<fft::alg::cpu::fftwc> {
  static constexpr std::string_view value = "FFTW3";
};

template <> struct fft_trait_name<fft::alg::cpu::edpc_first> {
  static constexpr std::string_view value = "EDP DIT Out of Place";
};

template <> struct fft_trait_name<fft::alg::cpu::edpc_second> {
  static constexpr std::string_view value = "EDP DIT In Place";
};

template <> struct fft_trait_name<fft::alg::cpu::edpc_third> {
  static constexpr std::string_view value = "EDP DIT In Place Out of cache";
};

} // namespace fft

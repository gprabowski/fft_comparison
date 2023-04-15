#pragma once

#include "cpu_fft.h"

#include <string_view>

namespace fft {

template <typename> struct fft_trait_name;

template <typename... Ts>
struct fft_trait_name<fft::alg::cpu::ibb_first<Ts...>> {
  static constexpr std::string_view value = "DIT NR from Inside FFT";
};

template <typename... Ts>
struct fft_trait_name<fft::alg::cpu::ibb_second<Ts...>> {
  static constexpr std::string_view value = "DIT RN from Inside FFT";
};

template <typename... Ts>
struct fft_trait_name<fft::alg::cpu::ibb_third<Ts...>> {
  static constexpr std::string_view value = "DIT NN from Inside FFT";
};

template <typename... Ts> struct fft_trait_name<fft::alg::cpu::fftw<Ts...>> {
  static constexpr std::string_view value = "FFTW3";
};

template <typename... Ts>
struct fft_trait_name<fft::alg::cpu::edp_first<Ts...>> {
  static constexpr std::string_view value = "EDP DIT Out of Place";
};

template <typename... Ts>
struct fft_trait_name<fft::alg::cpu::edp_second<Ts...>> {
  static constexpr std::string_view value = "EDP DIT In Place";
};

template <typename... Ts>
struct fft_trait_name<fft::alg::cpu::edp_third<Ts...>> {
  static constexpr std::string_view value = "EDP DIT In Place Out of cache";
};

} // namespace fft

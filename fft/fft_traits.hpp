#pragma once

#include "cpu_fft.h"

#include <string_view>

namespace fft {

enum class fft_order { NR, NN, RN };

enum class fft_type { DIT, DIF };

enum class twiddle_order { natural, reversed };

template <fft_order Order> struct order_owner {
  constexpr static fft_order value = Order;
};

template <fft_type Type> struct type_owner {
  constexpr static fft_type value = Type;
};

template <twiddle_order Order> struct twiddle_order_owner {
  constexpr static twiddle_order value = Order;
};

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

template <typename... Ts> struct fft_trait_name<fft::alg::cpu::edp_first<Ts...>> {
  static constexpr std::string_view value = "EDP First";
};

template <typename... Ts> struct fft_trait_name<fft::alg::cpu::edp_second<Ts...>> {
  static constexpr std::string_view value = "EDP Second";
};

template <typename> struct fft_trait_name;

template <typename> struct fft_trait_order;

template <typename... Ts>
struct fft_trait_order<fft::alg::cpu::ibb_first<Ts...>>
    : order_owner<fft_order::NR> {};

template <typename... Ts>
struct fft_trait_order<fft::alg::cpu::ibb_second<Ts...>>
    : order_owner<fft_order::RN> {};

template <typename... Ts>
struct fft_trait_order<fft::alg::cpu::fftw<Ts...>>
    : order_owner<fft_order::NN> {};

template <typename... Ts>
struct fft_trait_order<fft::alg::cpu::ibb_third<Ts...>>
    : order_owner<fft_order::NN> {};

template <typename... Ts>
struct fft_trait_order<fft::alg::cpu::edp_first<Ts...>>
    : order_owner<fft_order::NR> {};

template <typename... Ts>
struct fft_trait_order<fft::alg::cpu::edp_second<Ts...>>
    : order_owner<fft_order::NR> {};

template <typename> struct fft_trait_type;

template <typename... Ts>
struct fft_trait_type<fft::alg::cpu::ibb_first<Ts...>>
    : type_owner<fft_type::DIT> {};

template <typename... Ts>
struct fft_trait_type<fft::alg::cpu::ibb_second<Ts...>>
    : type_owner<fft_type::DIT> {};

template <typename... Ts>
struct fft_trait_type<fft::alg::cpu::ibb_third<Ts...>>
    : type_owner<fft_type::DIT> {};

template <typename... Ts>
struct fft_trait_type<fft::alg::cpu::edp_first<Ts...>>
    : type_owner<fft_type::DIT> {};

template <typename... Ts>
struct fft_trait_type<fft::alg::cpu::edp_second<Ts...>>
    : type_owner<fft_type::DIT> {};

template <typename... Ts>
struct fft_trait_type<fft::alg::cpu::fftw<Ts...>> : type_owner<fft_type::DIT> {
};

template <typename> struct fft_trait_twiddle_order;

template <typename... Ts>
struct fft_trait_twiddle_order<fft::alg::cpu::ibb_first<Ts...>>
    : twiddle_order_owner<twiddle_order::reversed> {};

template <typename... Ts>
struct fft_trait_twiddle_order<fft::alg::cpu::ibb_second<Ts...>>
    : twiddle_order_owner<twiddle_order::natural> {};

template <typename... Ts>
struct fft_trait_twiddle_order<fft::alg::cpu::fftw<Ts...>>
    : twiddle_order_owner<twiddle_order::natural> {};

template <typename... Ts>
struct fft_trait_twiddle_order<fft::alg::cpu::ibb_third<Ts...>>
    : twiddle_order_owner<twiddle_order::natural> {};

template <typename... Ts>
struct fft_trait_twiddle_order<fft::alg::cpu::edp_first<Ts...>>
    : twiddle_order_owner<twiddle_order::natural> {};

template <typename... Ts>
struct fft_trait_twiddle_order<fft::alg::cpu::edp_second<Ts...>>
    : twiddle_order_owner<twiddle_order::natural> {};
} // namespace fft

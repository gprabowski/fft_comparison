#pragma once

#include "cpu_fft.h"

namespace fft {
    enum class fft_order {
        NR, NN, RN
    };

    enum class fft_type {
        DIT, DIF
    };

    template<fft_order Order>
    struct order_owner {
       constexpr static fft_order value = Order;
    };

    template<fft_type Type>
    struct type_owner {
        constexpr static fft_type value = Type;
    };

    template<typename>
    struct fft_trait_order;

    template<typename...Ts>
    struct fft_trait_order<fft::alg::cpu::ibb_first<Ts...>> : order_owner<fft_order::NR> {};

    template<typename>
    struct fft_trait_type;

    template<typename...Ts>
    struct fft_trait_type<fft::alg::cpu::ibb_first<Ts...>> : type_owner<fft_type::DIT>{};
}

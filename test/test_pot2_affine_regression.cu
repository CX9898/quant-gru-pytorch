#include "quantize_ops_helper.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool runPot2Case() {
    const std::vector<int64_t> test_values = {0, 1, -1, 7, -7, 12345, -12345, 1LL << 30, -(1LL << 30)};
    const std::vector<int8_t> shifts = {0, 1, 3, 7, 12};

    for (const int64_t x : test_values) {
        for (const int8_t shift : shifts) {
            FixedPointScale pot_scale{1u, shift};
            const int64_t expected = rshift_round(x, shift);
            const int64_t got = rescale_round<true>(x, pot_scale);
            if (expected != got) {
                std::cerr << "[POT2] mismatch: x=" << x
                          << " shift=" << static_cast<int>(shift)
                          << " expected=" << expected
                          << " got=" << got << "\n";
                return false;
            }
        }
    }
    return true;
}

bool runAffineCase() {
    const std::vector<int64_t> test_values = {0, 3, -3, 127, -127, 4096, -4096, 1LL << 28, -(1LL << 28)};
    const std::vector<FixedPointScale> affine_scales = {
        {1u, 0},
        {1u, 7},
        {37u, 8},
        {511u, 12},
        {16384u, 15},
    };

    for (const int64_t x : test_values) {
        for (const auto s : affine_scales) {
            const int64_t expected = rshift_round(x * static_cast<int64_t>(s.multiplier), s.shift);
            const int64_t got = rescale_round<false>(x, s);
            if (expected != got) {
                std::cerr << "[Affine] mismatch: x=" << x
                          << " multiplier=" << s.multiplier
                          << " shift=" << static_cast<int>(s.shift)
                          << " expected=" << expected
                          << " got=" << got << "\n";
                return false;
            }
        }
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = (argc > 1) ? argv[1] : "all";

    if (mode == "pot2") {
        return runPot2Case() ? 0 : 1;
    }
    if (mode == "affine") {
        return runAffineCase() ? 0 : 1;
    }
    if (mode == "all") {
        return (runPot2Case() && runAffineCase()) ? 0 : 1;
    }

    std::cerr << "Unknown mode: " << mode << "\n";
    return 2;
}

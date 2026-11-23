#include <bitset>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

/// Prints the IEEE 754 bit fields (sign, exponent, unbiased E, and mantissa) of a float.
void print_bits(float x) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));

    const std::uint32_t sign = (bits >> 31) & 1u;
    const std::uint32_t exp  = (bits >> 23) & 0xFFu;
    const std::uint32_t frac = bits & 0x7FFFFFu;
    const int E = (exp == 0) ? -126 : static_cast<int>(exp) - 127;

    std::cout << "x = " << x << '\n'
              << "  sign = " << sign
              << " | exp = " << std::bitset<8>(exp) << " (raw=" << exp << ", E=" << E << ")"
              << " | mantissa = " << std::bitset<23>(frac) << '\n';
}

/// Converts an FP32 value to BF16 by keeping only its top 16 bits (truncation).
std::uint16_t fp32_to_bf16(float x) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

/// Converts a BF16 bit pattern back to FP32 by zero-filling the lower 16 mantissa bits.
float bf16_to_fp32(std::uint16_t b) {
    std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
    float x = 0.0f;
    std::memcpy(&x, &bits, sizeof(x));
    return x;
}

/// Measures max and average absolute round-trip BF16 error over an N x N random matrix in [-1, 1).
void measure_bf16_error(std::size_t N) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> data(N * N);
    for (float& x : data) {
        x = dist(rng);
    }

    float max_err = 0.0f;
    double sum_err = 0.0;
    for (float x : data) {
        const float back = bf16_to_fp32(fp32_to_bf16(x));
        const float err = std::fabs(x - back);
        sum_err += err;
        if (err > max_err) {
            max_err = err;
        }
    }

    const double avg_err = sum_err / static_cast<double>(data.size());
    std::cout << "\nBF16 error on " << N << "x" << N << " random matrix [-1, 1):\n"
              << "  max_abs_diff = " << max_err << '\n'
              << "  avg_abs_diff = " << avg_err << '\n';
}

/// Day 6 entry point: prints bit layouts for sample floats, tests BF16 round-trip, and measures error.
int main() {
    for (float x : {1.0f, 0.1f, -2.5f, 1e-40f}) {
        print_bits(x);
        const float back = bf16_to_fp32(fp32_to_bf16(x));
        std::cout << "  -> BF16 round-trip = " << back << '\n';
        print_bits(back);
    }

    measure_bf16_error(1024);
    return 0;
}

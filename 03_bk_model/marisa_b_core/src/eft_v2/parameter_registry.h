#ifndef MARISA_B_EFT_V2_PARAMETER_REGISTRY_H
#define MARISA_B_EFT_V2_PARAMETER_REGISTRY_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace marisa_b_eft_v2 {

enum class ParameterId : std::uint8_t {
    B1 = 0,
    B2,
    Gamma2,
    B3,
    Gamma2x,
    Gamma3,
    Gamma21,
    Gamma21x,
    Gamma211,
    Gamma22,
    Gamma31,
    BNabla2Delta,
    BNabla2Delta2,
    BNabla2G2,
    BGradDelta2,
    BGradT2,
    Pshot,
    A0Power,
    AshotResidual,
    A1Pure,
    BshotResidual,
    D2,
    DG2,
    DGamma3,
    Abar0Mixed,
    A3Mixed,
    A4Mixed,
    A5Mixed,
    Count
};

constexpr std::size_t kParameterCount = static_cast<std::size_t>(ParameterId::Count);
constexpr std::size_t kBiasParameterCount = 11;

enum class ParameterSector : std::uint8_t {
    Bias,
    Counterterm,
    Stochastic,
};

struct ParameterInfo {
    ParameterId id;
    std::string_view name;
    ParameterSector sector;
    int perturbative_order;
    bool shared_with_power;
};

const std::array<ParameterInfo, kParameterCount>& parameter_table();
const ParameterInfo& parameter_info(ParameterId id);
ParameterId parameter_id(std::string_view name);
bool is_bias_parameter(ParameterId id) noexcept;
std::string registry_canonical_string();
std::string registry_sha256();
std::string sha256_hex(std::string_view input);

}  // namespace marisa_b_eft_v2

#endif

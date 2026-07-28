#include "parameter_registry.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace marisa_b_eft_v2 {
namespace {

constexpr std::array<ParameterInfo, kParameterCount> kTable = {{
    {ParameterId::B1, "b1", ParameterSector::Bias, 1, true},
    {ParameterId::B2, "b2", ParameterSector::Bias, 2, true},
    {ParameterId::Gamma2, "gamma2", ParameterSector::Bias, 2, true},
    {ParameterId::B3, "b3", ParameterSector::Bias, 3, false},
    {ParameterId::Gamma2x, "gamma2x", ParameterSector::Bias, 3, false},
    {ParameterId::Gamma3, "gamma3", ParameterSector::Bias, 3, false},
    {ParameterId::Gamma21, "gamma21", ParameterSector::Bias, 3, true},
    {ParameterId::Gamma21x, "gamma21x", ParameterSector::Bias, 4, false},
    {ParameterId::Gamma211, "gamma211", ParameterSector::Bias, 4, false},
    {ParameterId::Gamma22, "gamma22", ParameterSector::Bias, 4, false},
    {ParameterId::Gamma31, "gamma31", ParameterSector::Bias, 4, false},
    {ParameterId::BNabla2Delta, "b_nabla2_delta", ParameterSector::Counterterm, 1, true},
    {ParameterId::BNabla2Delta2, "b_nabla2_delta2", ParameterSector::Counterterm, 2, false},
    {ParameterId::BNabla2G2, "b_nabla2_G2", ParameterSector::Counterterm, 2, false},
    {ParameterId::BGradDelta2, "b_grad_delta2", ParameterSector::Counterterm, 2, false},
    {ParameterId::BGradT2, "b_grad_t2", ParameterSector::Counterterm, 2, false},
    {ParameterId::Pshot, "Pshot", ParameterSector::Stochastic, 0, true},
    {ParameterId::A0Power, "a0_power", ParameterSector::Stochastic, 0, true},
    {ParameterId::AshotResidual, "Ashot_residual", ParameterSector::Stochastic, 0, false},
    {ParameterId::A1Pure, "a1_pure", ParameterSector::Stochastic, 0, false},
    {ParameterId::BshotResidual, "Bshot_residual", ParameterSector::Stochastic, 1, false},
    {ParameterId::D2, "d2", ParameterSector::Stochastic, 2, false},
    {ParameterId::DG2, "dG2", ParameterSector::Stochastic, 2, false},
    {ParameterId::DGamma3, "dGamma3", ParameterSector::Stochastic, 3, false},
    {ParameterId::Abar0Mixed, "abar0_mixed", ParameterSector::Stochastic, 1, false},
    {ParameterId::A3Mixed, "a3_mixed", ParameterSector::Stochastic, 1, false},
    {ParameterId::A4Mixed, "a4_mixed", ParameterSector::Stochastic, 1, false},
    {ParameterId::A5Mixed, "a5_mixed", ParameterSector::Stochastic, 1, false},
}};

constexpr std::array<std::uint32_t, 64> kSha256Constants = {{
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U
}};

std::uint32_t rotate_right(std::uint32_t value, int bits) {
    return (value >> bits) | (value << (32 - bits));
}

std::string sha256_impl(std::string_view input) {
    std::string data(input);
    const std::uint64_t bit_length = static_cast<std::uint64_t>(data.size()) * 8U;
    data.push_back(static_cast<char>(0x80));
    while ((data.size() % 64U) != 56U) data.push_back('\0');
    for (int shift = 56; shift >= 0; shift -= 8) {
        data.push_back(static_cast<char>((bit_length >> shift) & 0xffU));
    }
    std::array<std::uint32_t, 8> state = {{
        0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
        0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U}};
    for (std::size_t offset = 0; offset < data.size(); offset += 64U) {
        std::array<std::uint32_t, 64> words = {};
        for (int index = 0; index < 16; ++index) {
            words[index] =
                (static_cast<std::uint32_t>(static_cast<unsigned char>(data[offset + 4 * index])) << 24)
                | (static_cast<std::uint32_t>(static_cast<unsigned char>(data[offset + 4 * index + 1])) << 16)
                | (static_cast<std::uint32_t>(static_cast<unsigned char>(data[offset + 4 * index + 2])) << 8)
                | static_cast<std::uint32_t>(static_cast<unsigned char>(data[offset + 4 * index + 3]));
        }
        for (int index = 16; index < 64; ++index) {
            const std::uint32_t s0 = rotate_right(words[index - 15], 7)
                                     ^ rotate_right(words[index - 15], 18)
                                     ^ (words[index - 15] >> 3);
            const std::uint32_t s1 = rotate_right(words[index - 2], 17)
                                     ^ rotate_right(words[index - 2], 19)
                                     ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        std::uint32_t a=state[0],b=state[1],c=state[2],d=state[3];
        std::uint32_t e=state[4],f=state[5],g=state[6],h=state[7];
        for (int index = 0; index < 64; ++index) {
            const std::uint32_t s1 = rotate_right(e,6)^rotate_right(e,11)^rotate_right(e,25);
            const std::uint32_t choose = (e&f)^((~e)&g);
            const std::uint32_t temp1 = h+s1+choose+kSha256Constants[index]+words[index];
            const std::uint32_t s0 = rotate_right(a,2)^rotate_right(a,13)^rotate_right(a,22);
            const std::uint32_t majority = (a&b)^(a&c)^(b&c);
            const std::uint32_t temp2 = s0+majority;
            h=g; g=f; f=e; e=d+temp1; d=c; c=b; b=a; a=temp1+temp2;
        }
        state[0]+=a; state[1]+=b; state[2]+=c; state[3]+=d;
        state[4]+=e; state[5]+=f; state[6]+=g; state[7]+=h;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::uint32_t value : state) output << std::setw(8) << value;
    return output.str();
}

}  // namespace

const std::array<ParameterInfo, kParameterCount>& parameter_table() { return kTable; }

const ParameterInfo& parameter_info(ParameterId id) {
    const std::size_t index = static_cast<std::size_t>(id);
    if (index >= kTable.size() || kTable[index].id != id) {
        throw std::out_of_range("invalid EFT-v2 ParameterId");
    }
    return kTable[index];
}

ParameterId parameter_id(std::string_view name) {
    const auto match = std::find_if(kTable.begin(), kTable.end(), [name](const ParameterInfo& item) {
        return item.name == name;
    });
    if (match == kTable.end()) throw std::invalid_argument("unknown EFT-v2 parameter: " + std::string(name));
    return match->id;
}

bool is_bias_parameter(ParameterId id) noexcept {
    return static_cast<std::size_t>(id) < kBiasParameterCount;
}

std::string registry_canonical_string() {
    std::ostringstream output;
    for (const ParameterInfo& info : kTable) {
        output << static_cast<int>(info.id) << ':' << info.name << ':'
               << static_cast<int>(info.sector) << ':' << info.perturbative_order
               << ':' << (info.shared_with_power ? 1 : 0) << '\n';
    }
    return output.str();
}

std::string sha256_hex(std::string_view input) { return sha256_impl(input); }

std::string registry_sha256() { return sha256_hex(registry_canonical_string()); }

}  // namespace marisa_b_eft_v2

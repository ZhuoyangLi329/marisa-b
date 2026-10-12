/*
 * 程序纲要：
 * 1. 复用生产 stochastic.cpp 的原始实现，但把它直接包含进本诊断翻译单元，
 *    从而只在这个独立可执行文件内访问匿名命名空间中的逐外腿 mixed-cross
 *    primitive；生产 driver、生产 object 与生产 binary 均不改动。
 * 2. 用与 role-v3 生产驱动完全相同的 physical K1 形式记号 gamma31，分别构造
 *    common、output-only、shift-only 三种角色，并对 DensityOnly 与
 *    TiedDensityAndShift 两张 stochastic map 求值。
 * 3. 对 Bshot_residual 对应的第零个 mixed-cross 坐标，逐项写出 P22 bare、
 *    zero-lag subtraction、P13 bare、bias subtraction、UV-tail restoration
 *    以及重组值的一阶 physical-K1 系数；每项已乘上外部 K1，因而正好是
 *    bispectrum mixed-one-loop 中“单外腿”的贡献。
 * 4. 输出一条自描述 JSON；本程序不拟合 B000、不提交作业、不生成图像。
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

/*
 * 这是刻意的诊断构建方式：链接时排除 eft_v2_stochastic.o，因此不会产生重复
 * 公共符号；匿名命名空间 primitive 只在当前翻译单元可见。这样既能检查逐项
 * UV 账本，又不会改写正在 Slurm 作业中使用的生产二进制。
 */
#include "../eft_v2/stochastic.cpp"

namespace eft = marisa_b_eft_v2;
namespace hv1 = marisa_b_halo_v1;

namespace {

constexpr double kPngAmplitudeScale = 100.0;
constexpr eft::ParameterId kFormalPngMarker = eft::ParameterId::Gamma31;

class TabulatedColumn final : public PowerSpectrum {
public:
    TabulatedColumn(const std::string& path, int column) : column_(column) {
        if (column_ != 1 && column_ != 3) {
            throw std::invalid_argument("tabulated column must be 1 or 3");
        }
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open table: " + path);
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty() || line.front() == '#') continue;
            std::istringstream row(line);
            double k = 0.0;
            double power = 0.0;
            double ignored = 0.0;
            double transfer = 0.0;
            if (!(row >> k >> power >> ignored >> transfer)) {
                throw std::runtime_error("invalid four-column table row");
            }
            const double value = column_ == 1 ? power : transfer;
            if (!(k > 0.0 && value > 0.0)) {
                throw std::runtime_error("table values must be positive");
            }
            log_k_.push_back(std::log(k));
            log_value_.push_back(std::log(value));
        }
        if (log_k_.size() < 2) {
            throw std::runtime_error("table is too short");
        }
    }

    real Evaluate(real k) const override {
        if (!(k > 0.0)) return 0.0;
        const double x = std::log(k);
        std::size_t right = 1;
        if (x >= log_k_.back()) {
            right = log_k_.size() - 1;
        } else if (x > log_k_.front()) {
            right = static_cast<std::size_t>(
                std::upper_bound(log_k_.begin(), log_k_.end(), x)
                - log_k_.begin());
        }
        const std::size_t left = right - 1;
        const double fraction =
            (x - log_k_[left]) / (log_k_[right] - log_k_[left]);
        return std::exp(
            log_value_[left]
            + fraction * (log_value_[right] - log_value_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("TabulatedColumn has no Cosmology object");
    }

private:
    int column_ = 1;
    std::vector<double> log_k_;
    std::vector<double> log_value_;
};

/* 与生产 role-v3 driver 的 renormalized stochastic-density provider 相同。 */
class StochasticDensityKernelProvider final : public eft::FieldKernelProvider {
public:
    explicit StochasticDensityKernelProvider(
        const eft::FieldKernelProvider& base) : base_(base) {}

    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        const eft::SparsePolynomial full = base_.deterministic(momenta);
        eft::SparsePolynomial result;
        for (const eft::ParameterId id : {
                 eft::ParameterId::B1,
                 eft::ParameterId::B2,
                 eft::ParameterId::Gamma2,
                 eft::ParameterId::Gamma21}) {
            result += full.coefficient(eft::MonomialKey::variable(id))
                      * eft::SparsePolynomial::variable(id);
        }
        return result;
    }

    std::string_view name() const noexcept override {
        return "diagnostic_role_resolved_stochastic_density";
    }

private:
    const eft::FieldKernelProvider& base_;
};

/* 只改变 K1=b1+100*u/M(k)，与生产 driver 的物理响应约定逐字同构。 */
class SymbolicPhysicalPngK1Provider final : public eft::FieldKernelProvider {
public:
    SymbolicPhysicalPngK1Provider(
        const eft::FieldKernelProvider& base,
        const PowerSpectrum& transfer,
        double minimum_k)
        : base_(base), transfer_(transfer), minimum_k_(minimum_k) {}

    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        eft::SparsePolynomial result = base_.deterministic(momenta);
        for (const auto& term : result.terms()) {
            for (const eft::MonomialFactor& factor : term.first.factors()) {
                if (factor.id == kFormalPngMarker) {
                    throw std::runtime_error(
                        "formal PNG marker collides with physical kernel");
                }
            }
        }
        if (momenta.size() == 1) {
            const double k = hv1::norm(momenta.front());
            if (k >= minimum_k_) {
                const double transfer = transfer_(k);
                if (!(transfer > 0.0)) {
                    throw std::runtime_error("non-positive PNG transfer");
                }
                result += (kPngAmplitudeScale / transfer)
                          * eft::SparsePolynomial::variable(kFormalPngMarker);
            }
        }
        return result;
    }

    std::string_view name() const noexcept override {
        return "diagnostic_symbolic_physical_png_K1";
    }

private:
    const eft::FieldKernelProvider& base_;
    const PowerSpectrum& transfer_;
    double minimum_k_ = 0.0;
};

int parse_int(const char* text, const std::string& name) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != std::string(text).size() || value < 1) {
        throw std::invalid_argument("invalid positive integer for " + name);
    }
    return value;
}

double parse_double(const char* text, const std::string& name) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != std::string(text).size() || !std::isfinite(value)) {
        throw std::invalid_argument("invalid finite number for " + name);
    }
    return value;
}

/* 精确删除形式记号并保留指定 u 次数的普通 bias 多项式。 */
eft::SparsePolynomial marker_coefficient(
    const eft::SparsePolynomial& input,
    std::size_t requested_power) {
    eft::SparsePolynomial result;
    for (const auto& term : input.terms()) {
        std::size_t marker_power = 0;
        eft::SparsePolynomial coefficient =
            eft::SparsePolynomial::constant(term.second);
        for (const eft::MonomialFactor& factor : term.first.factors()) {
            if (factor.id == kFormalPngMarker) {
                marker_power = static_cast<std::size_t>(factor.power);
                continue;
            }
            for (std::size_t power = 0; power < factor.power; ++power) {
                coefficient = coefficient
                    * eft::SparsePolynomial::variable(factor.id);
            }
        }
        if (marker_power == requested_power) result += coefficient;
    }
    return result;
}

void write_json_string(std::string_view value) {
    std::cout << '"';
    for (const unsigned char character : value) {
        if (character == '"') std::cout << "\\\"";
        else if (character == '\\') std::cout << "\\\\";
        else if (character == '\n') std::cout << "\\n";
        else std::cout << character;
    }
    std::cout << '"';
}

void write_polynomial(const eft::SparsePolynomial& polynomial) {
    std::cout << '{';
    bool first = true;
    for (const auto& term : polynomial.terms()) {
        if (!first) std::cout << ',';
        first = false;
        write_json_string(term.first.canonical_string());
        std::cout << ':' << term.second;
    }
    std::cout << '}';
}

struct WeightedComponents {
    eft::SparsePolynomial p22_bare;
    eft::SparsePolynomial p22_subtraction;
    eft::SparsePolynomial p22;
    eft::SparsePolynomial p13_bare;
    eft::SparsePolynomial p13_subtraction;
    eft::SparsePolynomial p13_tail;
    eft::SparsePolynomial p13;
    eft::SparsePolynomial total;
};

WeightedComponents weight_external(
    const eft::SparsePolynomial& external,
    const eft::MixedStochasticPowerTemplates& cross) {
    constexpr std::size_t bshot_direction = 0;
    return WeightedComponents{
        external * cross.P22_bare[bshot_direction],
        external * cross.P22_zero_lag_subtraction[bshot_direction],
        external * cross.P22[bshot_direction],
        external * cross.P13_bare[bshot_direction],
        external * cross.P13_bias_subtraction[bshot_direction],
        external * cross.P13_uv_tail_restoration[bshot_direction],
        external * cross.P13[bshot_direction],
        external * cross.value[bshot_direction]};
}

WeightedComponents subtract_components(
    const WeightedComponents& tied,
    const WeightedComponents& density) {
    return WeightedComponents{
        tied.p22_bare - density.p22_bare,
        tied.p22_subtraction - density.p22_subtraction,
        tied.p22 - density.p22,
        tied.p13_bare - density.p13_bare,
        tied.p13_subtraction - density.p13_subtraction,
        tied.p13_tail - density.p13_tail,
        tied.p13 - density.p13,
        tied.total - density.total};
}

void write_components(const WeightedComponents& value) {
    const auto write = [](const char* name,
                          const eft::SparsePolynomial& polynomial,
                          bool first) {
        if (!first) std::cout << ',';
        write_json_string(name);
        std::cout << ':';
        write_polynomial(marker_coefficient(polynomial, 1));
    };
    std::cout << '{';
    write("P22_bare", value.p22_bare, true);
    write("P22_zero_lag_subtraction", value.p22_subtraction, false);
    write("P22", value.p22, false);
    write("P13_bare", value.p13_bare, false);
    write("P13_bias_subtraction", value.p13_subtraction, false);
    write("P13_uv_tail_restoration", value.p13_tail, false);
    write("P13", value.p13, false);
    write("total", value.total, false);
    std::cout << '}';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 18) {
            std::cerr
                << "usage: marisa_b_stochastic_k1_p13_components_diagnostic "
                << "POWER_TABLE PNG_TABLE KX KY KZ NR NMU NPHI QMIN QMAX "
                << "UV_TAIL_KMAX R_SMOOTH B_REC CELL_SIZE KMIN "
                << "ASYMPTOTIC_FACTOR MAX_LOG_RADIAL_RATIO\n";
            return 2;
        }
        const std::string power_path = argv[1];
        const std::string transfer_path = argv[2];
        const eft::Vec3 external{
            parse_double(argv[3], "kx"),
            parse_double(argv[4], "ky"),
            parse_double(argv[5], "kz")};

        eft::StochasticIntegrationConfig integration;
        integration.n_radial = parse_int(argv[6], "radial order");
        integration.n_mu = parse_int(argv[7], "mu order");
        integration.n_phi = parse_int(argv[8], "phi order");
        integration.qmin = parse_double(argv[9], "qmin");
        integration.qmax = parse_double(argv[10], "qmax");
        integration.restore_p13_uv_tail = true;
        integration.uv_tail_kmax = parse_double(argv[11], "UV tail kmax");

        hv1::ReconstructionConfig reconstruction;
        reconstruction.enabled = true;
        reconstruction.smoothing_radius = parse_double(argv[12], "R smooth");
        reconstruction.bias_recon = parse_double(argv[13], "b rec");
        reconstruction.cell_size = parse_double(argv[14], "cell size");
        const double minimum_k = parse_double(argv[15], "minimum k");
        /* argv[16] 是显式 asymptotic factor，便于后续独立扫描 32/64/128。 */
        integration.asymptotic_factor =
            parse_double(argv[16], "asymptotic factor");
        const double maximum_log_radial_ratio =
            parse_double(argv[17], "maximum log-radial ratio");
        integration.maximum_log_p13_radial_interval =
            maximum_log_radial_ratio > 1.0
            ? std::log(maximum_log_radial_ratio)
            : 0.0;

        if (!(hv1::norm(external) > 0.0)
            || !(integration.qmin > 0.0)
            || !(integration.qmax > integration.qmin)
            || !(integration.uv_tail_kmax > integration.qmax)
            || !(reconstruction.smoothing_radius > 0.0)
            || !(reconstruction.bias_recon > 0.0)
            || reconstruction.cell_size < 0.0
            || !(minimum_k > 0.0)
            || !(maximum_log_radial_ratio >= 1.0)
            || !(integration.asymptotic_factor > 1.0)) {
            throw std::invalid_argument("diagnostic domain contract failed");
        }

        const TabulatedColumn power(power_path, 1);
        const TabulatedColumn transfer(transfer_path, 3);
        const eft::EftBiasKernelProvider base;
        const StochasticDensityKernelProvider density_base(base);
        const SymbolicPhysicalPngK1Provider symbolic_output(
            base, transfer, minimum_k);
        const SymbolicPhysicalPngK1Provider symbolic_density(
            density_base, transfer, minimum_k);
        const SymbolicPhysicalPngK1Provider symbolic_shift(
            base, transfer, minimum_k);

        struct Role {
            const char* name;
            const eft::FieldKernelProvider& external;
            const eft::FieldKernelProvider& density;
            const eft::FieldKernelProvider& shift;
        };
        const std::array<Role, 3> roles{{
            {"common", symbolic_output, symbolic_density, symbolic_shift},
            {"output_only", symbolic_output, symbolic_density, base},
            {"shift_only", base, density_base, symbolic_shift}}};

        std::cout << std::setprecision(17)
                  << "{\"schema\":\"marisa-b-stochastic-k1-p13-components-diagnostic-v1\","
                  << "\"external\":[" << external.x << ',' << external.y << ','
                  << external.z << "],\"orders\":[" << integration.n_radial
                  << ',' << integration.n_mu << ',' << integration.n_phi
                  << "],\"q_range\":[" << integration.qmin << ','
                  << integration.qmax << "],\"uv_tail_kmax\":"
                  << integration.uv_tail_kmax
                  << ",\"asymptotic_factor\":"
                  << integration.asymptotic_factor
                  << ",\"maximum_log_radial_ratio\":"
                  << maximum_log_radial_ratio
                  << ",\"response_power\":1,\"roles\":{";

        bool first_role = true;
        for (const Role& role : roles) {
            const eft::SparsePolynomial external_linear =
                role.external.deterministic({external});
            const auto density_cross =
                eft::reconstructed_mixed_cross_power_impl(
                    power, external, role.density, role.shift,
                    reconstruction, integration,
                    eft::ReconstructedStochasticMap::DensityOnly);
            const auto tied_cross =
                eft::reconstructed_mixed_cross_power_impl(
                    power, external, role.density, role.shift,
                    reconstruction, integration,
                    eft::ReconstructedStochasticMap::TiedDensityAndShift);
            const WeightedComponents density =
                weight_external(external_linear, density_cross);
            const WeightedComponents tied =
                weight_external(external_linear, tied_cross);
            const WeightedComponents noisy =
                subtract_components(tied, density);

            if (!first_role) std::cout << ',';
            first_role = false;
            write_json_string(role.name);
            std::cout << ":{\"density_only\":";
            write_components(density);
            std::cout << ",\"tied\":";
            write_components(tied);
            std::cout << ",\"noisy_shift\":";
            write_components(noisy);
            std::cout << '}';
        }
        std::cout << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stochastic K1 P13 component diagnostic failed: "
                  << error.what() << '\n';
        return 1;
    }
}

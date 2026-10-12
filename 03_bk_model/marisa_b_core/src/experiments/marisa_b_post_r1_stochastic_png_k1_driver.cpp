/*
 * 程序纲要：
 * 1. 在不改动 MARISA-B 参数 registry 的前提下，把四阶 bias 坐标 gamma31
 *    临时当作“形式 PNG 振幅”记号 u，并在每次调用底层 kernel 前验证原始
 *    三阶以内 stochastic one-loop kernel 中绝不含 gamma31。
 * 2. 一次完成 common、output-only、shift-only 三种物理 K1 角色的壳层积分；
 *    积分结果仍是 SparsePolynomial，因此可按 gamma31 的幂精确拆出 u^0...u^6
 *    系数，而不再对七个振幅及一个离网振幅重复同一组三维积分。
 * 3. 逐项重组形式多项式、检查 u=0 的 legacy Gaussian 闭合及一阶角色和式，
 *    最后保持旧分析器所需的 degree-indexed JSON 结构。形式记号在写出前会被
 *    完全剥离，既不会进入参数表，也不会成为拟合参数。
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "PowerSpectrum.h"
#include "field_kernel_provider.h"
#include "lattice_shell_rule.h"
#include "parameter_registry.h"
#include "stochastic.h"

namespace eft=marisa_b_eft_v2;
namespace hv1=marisa_b_halo_v1;
namespace shell=marisa_b_shell_v1;

namespace {

constexpr std::size_t kPngPolynomialDegree=6;
constexpr std::size_t kPngSampleCount=kPngPolynomialDegree+1;
constexpr double kPngAmplitudeScale=100.0;
constexpr double kNumericValidationOffGrid=3.304348292242871;
constexpr std::array<double,8> kNumericValidationAmplitudes={{
    -3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,kNumericValidationOffGrid}};
/*
 * gamma31 的 perturbative order 是 4，而本驱动只调用至三阶 field kernel。
 * 它因此可作为一次性形式记号；SymbolicPhysicalPngK1Provider 仍会在运行时
 * 检查 wrapped provider 的每个返回值，若未来代码令 gamma31 真正进入该
 * sector，程序会立即失败，绝不会把物理 bias 与形式振幅混在一起。
 */
constexpr eft::ParameterId kFormalPngMarker=eft::ParameterId::Gamma31;

std::string file_sha256(const std::string& path) {
    std::ifstream input(path,std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot hash provenance input: "+path);
    }
    const std::string payload{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    return eft::sha256_hex(payload);
}

class TabulatedColumn final:public PowerSpectrum {
public:
    TabulatedColumn(const std::string& path,int column):column_(column) {
        if (column_!=1 &&column_!=3) {
            throw std::invalid_argument("tabulated column must be 1 or 3");
        }
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open table: "+path);
        std::string line;
        while (std::getline(input,line)) {
            if (line.empty() ||line[0]=='#') continue;
            std::istringstream row(line);
            double k=0.0;
            double power=0.0;
            double ignored=0.0;
            double transfer=0.0;
            if (!(row>>k>>power>>ignored>>transfer)) {
                throw std::runtime_error("invalid four-column table row");
            }
            const double value=column_==1?power:transfer;
            if (!(k>0.0 &&value>0.0)) {
                throw std::runtime_error("table values must be positive");
            }
            log_k_.push_back(std::log(k));
            log_value_.push_back(std::log(value));
        }
        if (log_k_.size()<2) throw std::runtime_error("table is too short");
    }

    real Evaluate(real k) const override {
        if (!(k>0.0)) return 0.0;
        const double x=std::log(k);
        std::size_t right=1;
        if (x>=log_k_.back()) {
            right=log_k_.size()-1;
        } else if (x>log_k_.front()) {
            right=static_cast<std::size_t>(
                std::upper_bound(log_k_.begin(),log_k_.end(),x)
                -log_k_.begin());
        }
        const std::size_t left=right-1;
        const double fraction=
            (x-log_k_[left])/(log_k_[right]-log_k_[left]);
        return std::exp(
            log_value_[left]
            +fraction*(log_value_[right]-log_value_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("TabulatedColumn has no Cosmology object");
    }

private:
    int column_;
    std::vector<double> log_k_;
    std::vector<double> log_value_;
};

/*
 * This is the Gaussian density-side provider used by the stochastic sector.
 * It deliberately retains only the renormalized bias directions that were
 * already present in the released stochastic templates.  A finite physical
 * K1 wrapper is applied outside this class, so its constant a/M(k) term is
 * not lost when the registered bias monomials are projected.
 */
class StochasticDensityKernelProvider final:
    public eft::FieldKernelProvider {
public:
    explicit StochasticDensityKernelProvider(
        const eft::FieldKernelProvider& base):base_(base) {}

    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        const eft::SparsePolynomial full=base_.deterministic(momenta);
        eft::SparsePolynomial result;
        for (const eft::ParameterId id:{
                 eft::ParameterId::B1,
                 eft::ParameterId::B2,
                 eft::ParameterId::Gamma2,
                 eft::ParameterId::Gamma21}) {
            result+=full.coefficient(eft::MonomialKey::variable(id))
                    *eft::SparsePolynomial::variable(id);
        }
        return result;
    }

    std::string_view name() const noexcept override {
        return "eft_v2_role_resolved_stochastic_density";
    }

private:
    const eft::FieldKernelProvider& base_;
};

/*
 * 只改变物理线性 kernel K1=b1+100*u/M(k)。这里的 u 由 gamma31 这个
 * 受保护的内部形式记号承载；输出前会按其幂拆分并彻底删除该记号。
 */
class SymbolicPhysicalPngK1Provider final:public eft::FieldKernelProvider {
public:
    SymbolicPhysicalPngK1Provider(
        const eft::FieldKernelProvider& base,
        const PowerSpectrum& transfer,
        double minimum_k):
        base_(base),transfer_(transfer),minimum_k_(minimum_k) {}

    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        eft::SparsePolynomial result=base_.deterministic(momenta);
        for (const auto& term:result.terms()) {
            for (const eft::MonomialFactor& factor:term.first.factors()) {
                if (factor.id==kFormalPngMarker) {
                    throw std::runtime_error(
                        "formal PNG marker collides with a physical kernel term");
                }
            }
        }
        if (momenta.size()==1) {
            const double k=hv1::norm(momenta.front());
            if (k>=minimum_k_) {
                const double transfer=transfer_(k);
                if (!(transfer>0.0)) {
                    throw std::runtime_error("non-positive PNG transfer");
                }
                result+=(kPngAmplitudeScale/transfer)
                        *eft::SparsePolynomial::variable(kFormalPngMarker);
            }
        }
        return result;
    }

    std::string_view name() const noexcept override {
        return "eft_v2_halo_with_symbolic_physical_png_K1";
    }

private:
    const eft::FieldKernelProvider& base_;
    const PowerSpectrum& transfer_;
    double minimum_k_;
};

/*
 * 旧式有限振幅 provider 只服务于显式 opt-in 的 canary 回归。生产任务默认
 * 从不构造它；设置 MARISA_B_STOCHASTIC_PNG_NUMERIC_VALIDATION=1 后，程序会
 * 对七个插值点和一个离网点重跑旧算法，并逐项核对单次符号积分的重组值。
 */
class NumericValidationPhysicalPngK1Provider final:
    public eft::FieldKernelProvider {
public:
    NumericValidationPhysicalPngK1Provider(
        const eft::FieldKernelProvider& base,
        const PowerSpectrum& transfer,
        double minimum_k,
        double normalized_amplitude):
        base_(base),transfer_(transfer),minimum_k_(minimum_k),
        physical_amplitude_(kPngAmplitudeScale*normalized_amplitude) {}

    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        eft::SparsePolynomial result=base_.deterministic(momenta);
        if (momenta.size()==1) {
            const double k=hv1::norm(momenta.front());
            if (k>=minimum_k_) {
                const double transfer=transfer_(k);
                if (!(transfer>0.0)) {
                    throw std::runtime_error("non-positive PNG transfer");
                }
                result+=eft::SparsePolynomial::constant(
                    physical_amplitude_/transfer);
            }
        }
        return result;
    }

    std::string_view name() const noexcept override {
        return "eft_v2_numeric_validation_physical_png_K1";
    }

private:
    const eft::FieldKernelProvider& base_;
    const PowerSpectrum& transfer_;
    double minimum_k_;
    double physical_amplitude_;
};

int parse_int(const char* text,const std::string& name,int minimum=1) {
    std::size_t consumed=0;
    const int value=std::stoi(text,&consumed);
    if (consumed!=std::string(text).size() ||value<minimum) {
        throw std::invalid_argument("invalid integer for "+name);
    }
    return value;
}

double parse_double(const char* text,const std::string& name) {
    std::size_t consumed=0;
    const double value=std::stod(text,&consumed);
    if (consumed!=std::string(text).size() ||!std::isfinite(value)) {
        throw std::invalid_argument("invalid number for "+name);
    }
    return value;
}

std::vector<double> read_edges(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open radial edge file: "+path);
    std::vector<double> result;
    std::string line;
    while (std::getline(input,line)) {
        if (line.empty() ||line[0]=='#') continue;
        std::istringstream row(line);
        double value=0.0;
        if (!(row>>value)) throw std::runtime_error("invalid radial edge row");
        result.push_back(value);
    }
    if (result.size()!=16) {
        throw std::runtime_error("post R1 radial edge file must contain 16 values");
    }
    for (std::size_t index=1;index<result.size();++index) {
        if (!(result[index]>result[index-1])) {
            throw std::runtime_error("radial edges must be strictly increasing");
        }
    }
    return result;
}

void add_weighted(
    eft::StochasticTemplates& destination,
    const eft::StochasticTemplates& source,
    double weight) {
    for (std::size_t index=0;index<eft::kStochasticCount;++index) {
        destination.pure[index]+=weight*source.pure[index];
        destination.mixed_tree[index]+=weight*source.mixed_tree[index];
        destination.mixed_one_loop[index]+=
            weight*source.mixed_one_loop[index];
        destination.derivative_mixed[index]+=
            weight*source.derivative_mixed[index];
        destination.raw[index]+=weight*source.raw[index];
    }
    destination.bshot_bnabla2_cross+=
        weight*source.bshot_bnabla2_cross;
}

eft::StochasticTemplates add_templates(
    const eft::StochasticTemplates& left,
    const eft::StochasticTemplates& right) {
    eft::StochasticTemplates result;
    add_weighted(result,left,1.0);
    add_weighted(result,right,1.0);
    return result;
}

using StochasticSeries=
    std::array<eft::StochasticTemplates,kPngSampleCount>;
using NumericValidationSeries=
    std::array<eft::StochasticTemplates,kNumericValidationAmplitudes.size()>;

/*
 * 壳层节点互不依赖，可安全并行计算；结果先按 node index 保存，再严格按原始
 * 顺序做权重累加，因此不同线程数不会改变求和顺序。numeric 结构只在显式
 * canary 中分配，生产路径不会承担其内存或计算成本。
 */
struct SymbolicNodeEvaluation {
    eft::ReconstructedStochasticTemplates common;
    eft::ReconstructedStochasticTemplates output_only;
    eft::ReconstructedStochasticTemplates shift_only;
    eft::ReconstructedStochasticTemplates legacy;
};

struct NumericValidationNodeEvaluation {
    std::array<
        eft::ReconstructedStochasticTemplates,
        kNumericValidationAmplitudes.size()> common;
    std::array<
        eft::ReconstructedStochasticTemplates,
        kNumericValidationAmplitudes.size()> output_only;
    std::array<
        eft::ReconstructedStochasticTemplates,
        kNumericValidationAmplitudes.size()> shift_only;
};

eft::StochasticTemplates recompose_series(
    const StochasticSeries& series,double normalized_amplitude) {
    eft::StochasticTemplates result;
    double amplitude_power=1.0;
    for (const auto& coefficient:series) {
        add_weighted(result,coefficient,amplitude_power);
        amplitude_power*=normalized_amplitude;
    }
    return result;
}

struct PolynomialError {
    double maximum_absolute=0.0;
    double maximum_scale=0.0;

    double relative() const {
        return maximum_absolute/std::max(maximum_scale,1.0e-300);
    }
};

void update_error(
    PolynomialError& error,
    const eft::SparsePolynomial& actual,
    const eft::SparsePolynomial& expected) {
    const eft::SparsePolynomial difference=actual-expected;
    for (const auto& term:difference.terms()) {
        error.maximum_absolute=std::max(
            error.maximum_absolute,std::fabs(term.second));
    }
    for (const auto& term:actual.terms()) {
        error.maximum_scale=std::max(
            error.maximum_scale,std::fabs(term.second));
    }
    for (const auto& term:expected.terms()) {
        error.maximum_scale=std::max(
            error.maximum_scale,std::fabs(term.second));
    }
}

PolynomialError stochastic_error(
    const eft::StochasticTemplates& actual,
    const eft::StochasticTemplates& expected) {
    PolynomialError error;
    for (std::size_t index=0;index<eft::kStochasticCount;++index) {
        update_error(error,actual.pure[index],expected.pure[index]);
        update_error(
            error,actual.mixed_tree[index],expected.mixed_tree[index]);
        update_error(
            error,actual.mixed_one_loop[index],
            expected.mixed_one_loop[index]);
        update_error(
            error,actual.derivative_mixed[index],
            expected.derivative_mixed[index]);
        update_error(error,actual.raw[index],expected.raw[index]);
    }
    update_error(
        error,actual.bshot_bnabla2_cross,
        expected.bshot_bnabla2_cross);
    return error;
}

void merge_error(
    PolynomialError& destination,
    const PolynomialError& source) {
    destination.maximum_absolute=std::max(
        destination.maximum_absolute,source.maximum_absolute);
    destination.maximum_scale=std::max(
        destination.maximum_scale,source.maximum_scale);
}

/*
 * 把一个含形式记号 u 的稀疏多项式精确拆成 sum_p c_p u^p。
 * MonomialKey 没有公开的“删除 factor”修改器，所以这里从公开 factors()
 * 重新构造 c_p；同时把 c_p*u^p 重组并与输入逐项比较，作为每个生产 bin
 * 都会执行的符号闭合。任何超过 degree six 的项都会显式报错。
 */
std::array<eft::SparsePolynomial,kPngSampleCount> split_formal_marker(
    const eft::SparsePolynomial& input,
    PolynomialError& recomposition_error) {
    std::array<eft::SparsePolynomial,kPngSampleCount> result;
    eft::SparsePolynomial recomposed;
    for (const auto& term:input.terms()) {
        std::size_t marker_power=0;
        eft::SparsePolynomial coefficient=
            eft::SparsePolynomial::constant(term.second);
        for (const eft::MonomialFactor& factor:term.first.factors()) {
            if (factor.id==kFormalPngMarker) {
                marker_power=static_cast<std::size_t>(factor.power);
                continue;
            }
            for (std::size_t power=0;power<factor.power;++power) {
                coefficient=coefficient
                    *eft::SparsePolynomial::variable(factor.id);
            }
        }
        if (marker_power>kPngPolynomialDegree) {
            throw std::runtime_error(
                "formal PNG polynomial exceeds declared degree six");
        }
        result[marker_power]+=coefficient;
        eft::SparsePolynomial marker_factor=
            eft::SparsePolynomial::constant(1.0);
        for (std::size_t power=0;power<marker_power;++power) {
            marker_factor=marker_factor
                *eft::SparsePolynomial::variable(kFormalPngMarker);
        }
        recomposed+=coefficient*marker_factor;
    }
    update_error(recomposition_error,input,recomposed);
    return result;
}

/*
 * 对 StochasticTemplates 的所有物理 pieces 使用同一拆分规则。返回结构仍与
 * 旧 seven-point driver 一致：数组下标就是 normalized amplitude u 的幂。
 */
StochasticSeries split_formal_marker_series(
    const eft::StochasticTemplates& symbolic,
    PolynomialError& recomposition_error) {
    StochasticSeries result;
    for (std::size_t index=0;index<eft::kStochasticCount;++index) {
        const auto pure=split_formal_marker(
            symbolic.pure[index],recomposition_error);
        const auto mixed_tree=split_formal_marker(
            symbolic.mixed_tree[index],recomposition_error);
        const auto mixed_one_loop=split_formal_marker(
            symbolic.mixed_one_loop[index],recomposition_error);
        const auto derivative_mixed=split_formal_marker(
            symbolic.derivative_mixed[index],recomposition_error);
        const auto raw=split_formal_marker(
            symbolic.raw[index],recomposition_error);
        for (std::size_t power=0;power<kPngSampleCount;++power) {
            result[power].pure[index]=pure[power];
            result[power].mixed_tree[index]=mixed_tree[power];
            result[power].mixed_one_loop[index]=mixed_one_loop[power];
            result[power].derivative_mixed[index]=derivative_mixed[power];
            result[power].raw[index]=raw[power];
        }
    }
    const auto cross=split_formal_marker(
        symbolic.bshot_bnabla2_cross,recomposition_error);
    for (std::size_t power=0;power<kPngSampleCount;++power) {
        result[power].bshot_bnabla2_cross=cross[power];
    }
    return result;
}

void write_json_string(const std::string& value) {
    std::cout<<'"';
    for (const unsigned char character:value) {
        switch (character) {
            case '"': std::cout<<"\\\""; break;
            case '\\': std::cout<<"\\\\"; break;
            case '\n': std::cout<<"\\n"; break;
            case '\r': std::cout<<"\\r"; break;
            case '\t': std::cout<<"\\t"; break;
            default:
                if (character<0x20) {
                    const auto flags=std::cout.flags();
                    const char fill=std::cout.fill();
                    std::cout<<"\\u"<<std::hex<<std::setw(4)
                             <<std::setfill('0')<<static_cast<int>(character);
                    std::cout.flags(flags);
                    std::cout.fill(fill);
                } else {
                    std::cout<<character;
                }
        }
    }
    std::cout<<'"';
}

void write_polynomial(const eft::SparsePolynomial& polynomial) {
    std::cout<<'{';
    bool first=true;
    for (const auto& term:polynomial.terms()) {
        if (!first) std::cout<<',';
        first=false;
        write_json_string(term.first.canonical_string());
        std::cout<<':'<<term.second;
    }
    std::cout<<'}';
}

void write_component(
    const std::array<eft::SparsePolynomial,eft::kStochasticCount>& values) {
    std::cout<<'{';
    for (std::size_t index=0;index<values.size();++index) {
        if (index>0) std::cout<<',';
        write_json_string(std::string(eft::parameter_info(
            eft::stochastic_parameter_ids()[index]).name));
        std::cout<<':';
        write_polynomial(values[index]);
    }
    std::cout<<'}';
}

void write_stochastic(const eft::StochasticTemplates& value) {
    std::cout<<"{\"pure\":";
    write_component(value.pure);
    std::cout<<",\"mixed_tree\":";
    write_component(value.mixed_tree);
    std::cout<<",\"mixed_one_loop\":";
    write_component(value.mixed_one_loop);
    std::cout<<",\"derivative_mixed\":";
    write_component(value.derivative_mixed);
    std::cout<<",\"raw\":";
    write_component(value.raw);
    std::cout<<",\"bshot_bnabla2_cross\":";
    write_polynomial(value.bshot_bnabla2_cross);
    std::cout<<'}';
}

void write_stochastic_series(const StochasticSeries& value) {
    std::cout<<'{';
    for (std::size_t power=0;power<value.size();++power) {
        if (power>0) std::cout<<',';
        write_json_string(std::to_string(power));
        std::cout<<':';
        write_stochastic(value[power]);
    }
    std::cout<<'}';
}

void write_noise_map_series(
    const StochasticSeries& tied,
    const StochasticSeries& density_only,
    const StochasticSeries& noisy_shift) {
    std::cout<<"{\"tied\":";
    write_stochastic_series(tied);
    std::cout<<",\"density_only\":";
    write_stochastic_series(density_only);
    std::cout<<",\"noisy_shift\":";
    write_stochastic_series(noisy_shift);
    std::cout<<'}';
}

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=22) {
            std::cerr
                <<"usage: marisa_b_post_r1_stochastic_png_k1_driver "
                <<"POWER_TABLE PNG_TABLE EDGE_FILE RADIAL_STOP START STOP "
                <<"SHELL_NRAD SHELL_NMU N_ALPHA N_COS_BETA N_GAMMA "
                <<"STOCH_NRAD STOCH_NMU STOCH_NPHI QMIN QMAX UV_TAIL_KMAX "
                <<"R_SMOOTH B_REC CELL_SIZE KMIN\n";
            return 2;
        }
        const std::string power_path=argv[1];
        const std::string transfer_path=argv[2];
        const std::string edge_path=argv[3];
        const int radial_stop=parse_int(argv[4],"radial stop");
        const int start=parse_int(argv[5],"start",0);
        const int stop=parse_int(argv[6],"stop");
        if (radial_stop>15 ||start<0 ||stop<=start ||stop>120) {
            throw std::invalid_argument(
                "require radial_stop<=15 and 0<=start<stop<=120");
        }

        eft::HaarOrientedLatticeShellRuleConfig shell_config;
        shell_config.exact.radial_order=parse_int(argv[7],"shell radial order");
        shell_config.exact.angular_order=parse_int(argv[8],"shell angular order");
        shell_config.exact.fft_box_size=1000.0;
        shell_config.exact.fft_mesh_size=256;
        shell_config.n_alpha=parse_int(argv[9],"alpha order");
        shell_config.n_cos_beta=parse_int(argv[10],"cos-beta order");
        shell_config.n_gamma=parse_int(argv[11],"gamma order");

        eft::StochasticIntegrationConfig stochastic;
        stochastic.n_radial=parse_int(argv[12],"stochastic radial order");
        stochastic.n_mu=parse_int(argv[13],"stochastic mu order");
        stochastic.n_phi=parse_int(argv[14],"stochastic phi order");
        stochastic.qmin=parse_double(argv[15],"qmin");
        stochastic.qmax=parse_double(argv[16],"qmax");
        stochastic.restore_p13_uv_tail=true;
        stochastic.uv_tail_kmax=parse_double(argv[17],"UV tail kmax");

        hv1::ReconstructionConfig reconstruction;
        reconstruction.enabled=true;
        reconstruction.smoothing_radius=parse_double(argv[18],"smoothing radius");
        reconstruction.bias_recon=parse_double(argv[19],"reconstruction bias");
        reconstruction.cell_size=parse_double(argv[20],"cell size");
        const double minimum_k=parse_double(argv[21],"minimum PNG wavenumber");
        const char* numeric_validation_environment=
            std::getenv("MARISA_B_STOCHASTIC_PNG_NUMERIC_VALIDATION");
        const bool numeric_validation_enabled=
            numeric_validation_environment!=nullptr
            &&std::string(numeric_validation_environment)=="1";
        const int maximum_openmp_threads=omp_get_max_threads();
        if (!(stochastic.qmin>0.0)
            ||!(stochastic.qmax>stochastic.qmin)
            ||!(stochastic.uv_tail_kmax>stochastic.qmax)
            ||!(reconstruction.smoothing_radius>0.0)
            ||!(reconstruction.bias_recon>0.0)
            ||reconstruction.cell_size<0.0
            ||!(minimum_k>0.0)) {
            throw std::invalid_argument(
                "stochastic/reconstruction domain contract failed");
        }

        const TabulatedColumn power(power_path,1);
        const TabulatedColumn transfer(transfer_path,3);
        const std::vector<double> edges=read_edges(edge_path);
        const eft::EftBiasKernelProvider base;
        const StochasticDensityKernelProvider density_base(base);
        const SymbolicPhysicalPngK1Provider symbolic_output(
            base,transfer,minimum_k);
        const SymbolicPhysicalPngK1Provider symbolic_density(
            density_base,transfer,minimum_k);
        const SymbolicPhysicalPngK1Provider symbolic_shift(
            base,transfer,minimum_k);

        std::cout<<std::setprecision(17)
                 <<"{\"record\":\"header\","
                 <<"\"schema\":\"marisa-b-post-r1-stochastic-physical-png-k1-role-v3\"," 
                 <<"\"production_candidate\":false,"
                 <<"\"scope\":\"source-output-role-resolved physical-K1 response of already-retained reconstructed stochastic templates\","
                 <<"\"parameter_registry_modified\":false,"
                 <<"\"response_rule\":\"normalized-amplitude coefficient p multiplies (fNL*bphi/amplitude_scale)^p\","
                 <<"\"physical_K1_roles\":[\"common\",\"output_only\",\"shift_only\"],"
                 <<"\"role_contract\":{"
                 <<"\"common\":\"external output, internal stochastic-density and displacement-source K1 vary together\","
                 <<"\"output_only\":\"external output and internal stochastic-density K1 vary; displacement source stays Gaussian\","
                 <<"\"shift_only\":\"only displacement-source K1 varies\","
                 <<"\"final_standard_assignment\":\"same physical halo tracer ties output and source bphi; operational b_rec remains fixed\","
                 <<"\"bphi_rec_metadata_used_as_physical_bias\":false},"
                 <<"\"polynomial_degree\":"<<kPngPolynomialDegree<<','
                 <<"\"amplitude_scale\":"<<kPngAmplitudeScale<<','
                 <<"\"coefficient_generation\":\"single-pass exact SparsePolynomial formal-marker extraction\","
                 <<"\"formal_marker\":\"gamma31\","
                 <<"\"formal_marker_physical_kernel_order\":4,"
                 <<"\"maximum_kernel_order_in_scope\":3,"
                 <<"\"formal_marker_collision_runtime_guard\":true,"
                 <<"\"formal_marker_removed_before_output\":true,"
                 <<"\"repeated_finite_amplitude_integrations\":false,"
                 <<"\"shell_node_parallelism\":\"OpenMP compute then ordered serial reduction\","
                 <<"\"maximum_openmp_threads\":"
                 <<maximum_openmp_threads<<','
                 <<"\"numeric_finite_amplitude_validation_enabled\":"
                 <<(numeric_validation_enabled?"true":"false")<<','
                 <<"\"external_K1_product_rule\":true,"
                 <<"\"internal_density_K1_product_rule\":true,"
                 <<"\"internal_shift_K1_product_rule\":true,"
                 <<"\"overlap_contract\":{"
                 <<"\"already_released\":\"Bshot_residual mixed_tree physical-K1 response\","
                 <<"\"candidate_addition\":\"fitted Gaussian stochastic coordinates times amplitude response after subtracting already-released mixed_tree\"},"
                 <<"\"bin_range\":["<<start<<','<<stop<<"],"
                 <<"\"radial_stop\":"<<radial_stop<<','
                 <<"\"reconstruction\":{\"R\":"
                 <<reconstruction.smoothing_radius
                 <<",\"b_rec_h\":"<<reconstruction.bias_recon
                 <<",\"cell_size\":"<<reconstruction.cell_size
                 <<",\"kmin\":"<<minimum_k<<"},"
                 <<"\"shell_orders\":["
                 <<shell_config.exact.radial_order<<','
                 <<shell_config.exact.angular_order<<','
                 <<shell_config.n_alpha<<','
                 <<shell_config.n_cos_beta<<','
                 <<shell_config.n_gamma<<"],"
                 <<"\"stochastic_orders\":["
                 <<stochastic.n_radial<<','
                 <<stochastic.n_mu<<','
                 <<stochastic.n_phi<<"],"
                 <<"\"q_range\":["<<stochastic.qmin<<','
                 <<stochastic.qmax<<"],"
                 <<"\"uv_tail_kmax\":"<<stochastic.uv_tail_kmax<<','
                 <<"\"source_hashes\":{\"linear_power\":";
        write_json_string(file_sha256(power_path));
        std::cout<<",\"png_table\":";
        write_json_string(file_sha256(transfer_path));
        std::cout<<",\"edge_file\":";
        write_json_string(file_sha256(edge_path));
        std::cout<<",\"driver_executable\":";
        write_json_string(file_sha256(argv[0]));
        std::cout<<"},\"parameter_registry_sha256\":";
        write_json_string(eft::registry_sha256());
        std::cout<<"}\n";
        std::cout.flush();

        int flat=0;
        for (int first=0;first<15;++first) {
            for (int second=first;second<15;++second,++flat) {
                if (flat<start ||flat>=stop) continue;
                if (first>=radial_stop ||second>=radial_stop) continue;
                const shell::ShellBin bin{
                    edges[static_cast<std::size_t>(first)],
                    edges[static_cast<std::size_t>(first+1)],
                    edges[static_cast<std::size_t>(second)],
                    edges[static_cast<std::size_t>(second+1)]};
                const eft::HaarOrientedLatticeShellRule rule=
                    eft::make_haar_oriented_lattice_shell_rule(
                        bin,shell_config);
                /*
                 * 每个角色只积分一次，积分中的 gamma31 幂就是 normalized
                 * amplitude u 的幂；壳层平均与按幂取系数均为线性运算，顺序可交换。
                 */
                eft::StochasticTemplates common_tied_symbolic;
                eft::StochasticTemplates common_density_symbolic;
                eft::StochasticTemplates common_noisy_symbolic;
                eft::StochasticTemplates output_tied_symbolic;
                eft::StochasticTemplates output_density_symbolic;
                eft::StochasticTemplates output_noisy_symbolic;
                eft::StochasticTemplates shift_tied_symbolic;
                eft::StochasticTemplates shift_density_symbolic;
                eft::StochasticTemplates shift_noisy_symbolic;
                eft::StochasticTemplates legacy_tied;
                eft::StochasticTemplates legacy_density;
                eft::StochasticTemplates legacy_noisy;
                NumericValidationSeries numeric_common_tied;
                NumericValidationSeries numeric_common_density;
                NumericValidationSeries numeric_common_noisy;
                NumericValidationSeries numeric_output_tied;
                NumericValidationSeries numeric_output_density;
                NumericValidationSeries numeric_output_noisy;
                NumericValidationSeries numeric_shift_tied;
                NumericValidationSeries numeric_shift_density;
                NumericValidationSeries numeric_shift_noisy;
                std::vector<SymbolicNodeEvaluation> symbolic_nodes(
                    rule.nodes.size());
                std::vector<NumericValidationNodeEvaluation> numeric_nodes(
                    numeric_validation_enabled?rule.nodes.size():0);
                std::vector<std::string> node_errors(rule.nodes.size());
#pragma omp parallel for schedule(dynamic)
                for (std::int64_t node_index=0;
                     node_index<static_cast<std::int64_t>(rule.nodes.size());
                     ++node_index) {
                    try {
                        const shell::ShellNode& node=
                            rule.nodes[static_cast<std::size_t>(node_index)];
                        SymbolicNodeEvaluation& evaluation=
                            symbolic_nodes[static_cast<std::size_t>(node_index)];
                        evaluation.common=
                            eft::reconstructed_stochastic_bispectrum_role_decomposition(
                                power,node.closed_vectors,
                                symbolic_output,symbolic_density,
                                symbolic_shift,
                                reconstruction,0.30,stochastic,&power);
                        evaluation.output_only=
                            eft::reconstructed_stochastic_bispectrum_role_decomposition(
                                power,node.closed_vectors,
                                symbolic_output,symbolic_density,base,
                                reconstruction,0.30,stochastic,&power);
                        evaluation.shift_only=
                            eft::reconstructed_stochastic_bispectrum_role_decomposition(
                                power,node.closed_vectors,
                                base,density_base,symbolic_shift,
                                reconstruction,0.30,stochastic,&power);
                        evaluation.legacy=
                            eft::reconstructed_stochastic_bispectrum_decomposition(
                                power,node.closed_vectors,base,
                                reconstruction,0.30,stochastic,&power);
                        if (numeric_validation_enabled) {
                            NumericValidationNodeEvaluation& numeric=
                                numeric_nodes[
                                    static_cast<std::size_t>(node_index)];
                            for (std::size_t sample=0;
                                 sample<kNumericValidationAmplitudes.size();
                                 ++sample) {
                                const double amplitude=
                                    kNumericValidationAmplitudes[sample];
                                const NumericValidationPhysicalPngK1Provider
                                    numeric_output_provider(
                                        base,transfer,minimum_k,amplitude);
                                const NumericValidationPhysicalPngK1Provider
                                    numeric_density_provider(
                                        density_base,transfer,minimum_k,amplitude);
                                const NumericValidationPhysicalPngK1Provider
                                    numeric_shift_provider(
                                        base,transfer,minimum_k,amplitude);
                                numeric.common[sample]=
                                    eft::reconstructed_stochastic_bispectrum_role_decomposition(
                                        power,node.closed_vectors,
                                        numeric_output_provider,
                                        numeric_density_provider,
                                        numeric_shift_provider,
                                        reconstruction,0.30,stochastic,&power);
                                numeric.output_only[sample]=
                                    eft::reconstructed_stochastic_bispectrum_role_decomposition(
                                        power,node.closed_vectors,
                                        numeric_output_provider,
                                        numeric_density_provider,base,
                                        reconstruction,0.30,stochastic,&power);
                                numeric.shift_only[sample]=
                                    eft::reconstructed_stochastic_bispectrum_role_decomposition(
                                        power,node.closed_vectors,
                                        base,density_base,numeric_shift_provider,
                                        reconstruction,0.30,stochastic,&power);
                            }
                        }
                    } catch (const std::exception& error) {
                        node_errors[static_cast<std::size_t>(node_index)]=
                            error.what();
                    }
                }
                /*
                 * 并行区只写各自的 node slot；这里按 rule.nodes 的原始次序串行
                 * 归约，保证与旧驱动完全相同的浮点加法顺序和可复现性。
                 */
                for (std::size_t node_index=0;
                     node_index<rule.nodes.size();
                     ++node_index) {
                    if (!node_errors[node_index].empty()) {
                        throw std::runtime_error(
                            "shell node "+std::to_string(node_index)
                            +" failed: "+node_errors[node_index]);
                    }
                    const double weight=rule.nodes[node_index].weight;
                    const SymbolicNodeEvaluation& evaluation=
                        symbolic_nodes[node_index];
                    add_weighted(
                        common_tied_symbolic,evaluation.common.tied,weight);
                    add_weighted(
                        common_density_symbolic,
                        evaluation.common.density_only,weight);
                    add_weighted(
                        common_noisy_symbolic,
                        evaluation.common.noisy_shift,weight);
                    add_weighted(
                        output_tied_symbolic,
                        evaluation.output_only.tied,weight);
                    add_weighted(
                        output_density_symbolic,
                        evaluation.output_only.density_only,weight);
                    add_weighted(
                        output_noisy_symbolic,
                        evaluation.output_only.noisy_shift,weight);
                    add_weighted(
                        shift_tied_symbolic,
                        evaluation.shift_only.tied,weight);
                    add_weighted(
                        shift_density_symbolic,
                        evaluation.shift_only.density_only,weight);
                    add_weighted(
                        shift_noisy_symbolic,
                        evaluation.shift_only.noisy_shift,weight);
                    add_weighted(
                        legacy_tied,evaluation.legacy.tied,weight);
                    add_weighted(
                        legacy_density,
                        evaluation.legacy.density_only,weight);
                    add_weighted(
                        legacy_noisy,evaluation.legacy.noisy_shift,weight);
                    if (numeric_validation_enabled) {
                        const NumericValidationNodeEvaluation& numeric=
                            numeric_nodes[node_index];
                        for (std::size_t sample=0;
                             sample<kNumericValidationAmplitudes.size();
                             ++sample) {
                            add_weighted(
                                numeric_common_tied[sample],
                                numeric.common[sample].tied,weight);
                            add_weighted(
                                numeric_common_density[sample],
                                numeric.common[sample].density_only,weight);
                            add_weighted(
                                numeric_common_noisy[sample],
                                numeric.common[sample].noisy_shift,weight);
                            add_weighted(
                                numeric_output_tied[sample],
                                numeric.output_only[sample].tied,weight);
                            add_weighted(
                                numeric_output_density[sample],
                                numeric.output_only[sample].density_only,weight);
                            add_weighted(
                                numeric_output_noisy[sample],
                                numeric.output_only[sample].noisy_shift,weight);
                            add_weighted(
                                numeric_shift_tied[sample],
                                numeric.shift_only[sample].tied,weight);
                            add_weighted(
                                numeric_shift_density[sample],
                                numeric.shift_only[sample].density_only,weight);
                            add_weighted(
                                numeric_shift_noisy[sample],
                                numeric.shift_only[sample].noisy_shift,weight);
                        }
                    }
                }
                PolynomialError symbolic_recomposition_error;
                const StochasticSeries common_tied=split_formal_marker_series(
                    common_tied_symbolic,symbolic_recomposition_error);
                const StochasticSeries common_density=split_formal_marker_series(
                    common_density_symbolic,symbolic_recomposition_error);
                const StochasticSeries common_noisy=split_formal_marker_series(
                    common_noisy_symbolic,symbolic_recomposition_error);
                const StochasticSeries output_tied=split_formal_marker_series(
                    output_tied_symbolic,symbolic_recomposition_error);
                const StochasticSeries output_density=split_formal_marker_series(
                    output_density_symbolic,symbolic_recomposition_error);
                const StochasticSeries output_noisy=split_formal_marker_series(
                    output_noisy_symbolic,symbolic_recomposition_error);
                const StochasticSeries shift_tied=split_formal_marker_series(
                    shift_tied_symbolic,symbolic_recomposition_error);
                const StochasticSeries shift_density=split_formal_marker_series(
                    shift_density_symbolic,symbolic_recomposition_error);
                const StochasticSeries shift_noisy=split_formal_marker_series(
                    shift_noisy_symbolic,symbolic_recomposition_error);
                PolynomialError numeric_validation_error;
                if (numeric_validation_enabled) {
                    for (const auto& item:{
                             std::pair<const StochasticSeries*,
                                       const NumericValidationSeries*>{
                                 &common_tied,&numeric_common_tied},
                             std::pair<const StochasticSeries*,
                                       const NumericValidationSeries*>{
                                 &common_density,&numeric_common_density},
                             std::pair<const StochasticSeries*,
                                       const NumericValidationSeries*>{
                                 &common_noisy,&numeric_common_noisy},
                             std::pair<const StochasticSeries*,
                                       const NumericValidationSeries*>{
                                 &output_tied,&numeric_output_tied},
                             std::pair<const StochasticSeries*,
                                       const NumericValidationSeries*>{
                                 &output_density,&numeric_output_density},
                             std::pair<const StochasticSeries*,
                                       const NumericValidationSeries*>{
                                 &output_noisy,&numeric_output_noisy},
                             std::pair<const StochasticSeries*,
                                       const NumericValidationSeries*>{
                                 &shift_tied,&numeric_shift_tied},
                             std::pair<const StochasticSeries*,
                                       const NumericValidationSeries*>{
                                 &shift_density,&numeric_shift_density},
                             std::pair<const StochasticSeries*,
                                       const NumericValidationSeries*>{
                                 &shift_noisy,&numeric_shift_noisy}}) {
                        for (std::size_t sample=0;
                             sample<kNumericValidationAmplitudes.size();
                             ++sample) {
                            merge_error(
                                numeric_validation_error,
                                stochastic_error(
                                    recompose_series(
                                        *item.first,
                                        kNumericValidationAmplitudes[sample]),
                                    (*item.second)[sample]));
                        }
                    }
                }
                PolynomialError zero_role_error;
                for (const auto& pair:{
                         std::pair<const eft::StochasticTemplates*,
                                   const eft::StochasticTemplates*>{
                             &common_tied[0],&legacy_tied},
                         std::pair<const eft::StochasticTemplates*,
                                   const eft::StochasticTemplates*>{
                             &common_density[0],&legacy_density},
                         std::pair<const eft::StochasticTemplates*,
                                   const eft::StochasticTemplates*>{
                             &common_noisy[0],&legacy_noisy},
                         std::pair<const eft::StochasticTemplates*,
                                   const eft::StochasticTemplates*>{
                             &output_tied[0],&legacy_tied},
                         std::pair<const eft::StochasticTemplates*,
                                   const eft::StochasticTemplates*>{
                             &shift_tied[0],&legacy_tied}}) {
                    merge_error(
                        zero_role_error,
                        stochastic_error(*pair.first,*pair.second));
                }
                PolynomialError linear_role_error;
                for (const auto& item:{
                         std::array<const StochasticSeries*,3>{
                             &common_tied,&output_tied,&shift_tied},
                         std::array<const StochasticSeries*,3>{
                             &common_density,&output_density,&shift_density},
                         std::array<const StochasticSeries*,3>{
                             &common_noisy,&output_noisy,&shift_noisy}}) {
                    merge_error(
                        linear_role_error,
                        stochastic_error(
                            (*item[0])[1],
                            add_templates((*item[1])[1],(*item[2])[1])));
                }
                if (symbolic_recomposition_error.maximum_absolute>1.0e-10
                    &&symbolic_recomposition_error.relative()>2.0e-13) {
                    throw std::runtime_error(
                        "physical-K1 formal-marker extraction failed closure");
                }
                if (numeric_validation_enabled
                    &&numeric_validation_error.maximum_absolute>1.0e-6
                    &&numeric_validation_error.relative()>2.0e-11) {
                    throw std::runtime_error(
                        "symbolic physical-K1 result disagrees with finite-amplitude validation");
                }
                if (zero_role_error.maximum_absolute>1.0e-8
                    &&zero_role_error.relative()>2.0e-11) {
                    throw std::runtime_error(
                        "zero-amplitude role split does not reproduce legacy Gaussian templates");
                }
                if (linear_role_error.maximum_absolute>1.0e-8
                    &&linear_role_error.relative()>2.0e-11) {
                    throw std::runtime_error(
                        "linear common/output/shift K1 product rule failed closure");
                }
                std::cout<<"{\"record\":\"bin\",\"index\":"
                         <<flat<<",\"edges\":["
                         <<bin.k1_lower<<','<<bin.k1_upper<<','
                         <<bin.k2_lower<<','<<bin.k2_upper<<"],"
                         <<"\"shell_nodes\":"<<rule.nodes.size()<<','
                         <<"\"invariant_shell_nodes\":"
                         <<rule.diagnostics.invariant_nodes<<','
                         <<"\"orientation_nodes\":"
                         <<rule.diagnostics.orientation_nodes<<','
                         <<"\"zero_external_leg_pairs\":"
                         <<rule.diagnostics.exact.zero_external_leg_pairs<<','
                         <<"\"closing_zero_pairs\":"
                         <<rule.diagnostics.exact.closing_zero_pairs<<','
                         <<"\"valid_pair_fraction\":"
                         <<rule.diagnostics.exact.valid_pair_fraction<<','
                         <<"\"symbolic_recomposition_max_abs\":"
                         <<symbolic_recomposition_error.maximum_absolute<<','
                         <<"\"symbolic_recomposition_max_rel\":"
                         <<symbolic_recomposition_error.relative()<<','
                         <<"\"numeric_validation_max_abs\":"
                         <<numeric_validation_error.maximum_absolute<<','
                         <<"\"numeric_validation_max_rel\":"
                         <<numeric_validation_error.relative()<<','
                         <<"\"zero_role_legacy_max_abs\":"
                         <<zero_role_error.maximum_absolute<<','
                         <<"\"zero_role_legacy_max_rel\":"
                         <<zero_role_error.relative()<<','
                         <<"\"linear_role_sum_max_abs\":"
                         <<linear_role_error.maximum_absolute<<','
                         <<"\"linear_role_sum_max_rel\":"
                         <<linear_role_error.relative()<<','
                         <<"\"stochastic_role_coefficients\":{"
                         <<"\"common\":";
                write_noise_map_series(
                    common_tied,common_density,common_noisy);
                std::cout<<",\"output_only\":";
                write_noise_map_series(
                    output_tied,output_density,output_noisy);
                std::cout<<",\"shift_only\":";
                write_noise_map_series(
                    shift_tied,shift_density,shift_noisy);
                std::cout<<'}';
                std::cout<<"}\n";
                std::cout.flush();
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"post-R1 stochastic physical-PNG K1 driver failed: "
                 <<error.what()<<'\n';
        return 1;
    }
}

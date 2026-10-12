/*
 * EQ Delta=2 halo tree 的 strict-three-side exact-lattice projector。
 *
 * 代码大纲
 * ========
 * 1. 读取 k,P_L,P_phi,M 四列表，并用 log-log 线性插值提供 P_L/M；
 * 2. 在 L=1000 Mpc/h、mesh=256 的整数 FFT lattice 上枚举所有
 *    k_f<=|k|<=0.08 h/Mpc modes；
 * 3. 枚举闭合 ordered triplets，并只保留 shell_i<=shell_j<=shell_k，
 *    从而逐项匹配 M1 geometry 的固定 shell-role triangle counts；
 * 4. 对每个 exact Cartesian triangle 调用唯一 MARISA-B native Delta=2
 *    pre/post tree API，累计 Gaussian、scalar、spin-2、reconstruction 与
 *    stochastic raw bases；
 * 5. 按固定线程分块和固定线程编号归并，输出可由 Python 与 geometry NPZ
 *    精确审计的 JSON。
 *
 * 本程序不做 bin-center 近似、不调用 loop integration、不含 T_G/gain。
 */

#include "marisa_b_native.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "PowerSpectrum.h"

namespace native=marisa_b_native;

namespace {

constexpr double kPi=3.141592653589793238462643383279502884;
constexpr double kBoxSize=1000.0;
constexpr int kMeshSize=256;
constexpr int kShellCount=6;
constexpr double kMaximum=0.08;
constexpr int kMaximumIndex=12;
constexpr int kLookupSide=2*kMaximumIndex+1;
constexpr int kCombinationCount=56;

const std::array<double,kShellCount+1> kEdges{{
    2.0*kPi/kBoxSize,
    3.0*2.0*kPi/kBoxSize,
    5.0*2.0*kPi/kBoxSize,
    7.0*2.0*kPi/kBoxSize,
    9.0*2.0*kPi/kBoxSize,
    11.0*2.0*kPi/kBoxSize,
    kMaximum}};

double parse_double(const char* text,const std::string& label) {
    std::size_t consumed=0;
    const double value=std::stod(text,&consumed);
    if (consumed!=std::string(text).size()||!std::isfinite(value)) {
        throw std::invalid_argument("invalid number for "+label);
    }
    return value;
}

int parse_int(const char* text,const std::string& label) {
    std::size_t consumed=0;
    const int value=std::stoi(text,&consumed);
    if (consumed!=std::string(text).size()) {
        throw std::invalid_argument("invalid integer for "+label);
    }
    return value;
}

class LogColumnTable final:public PowerSpectrum {
public:
    LogColumnTable(
        const std::string& path,int target_column,int required_columns)
        :target_column_(target_column),required_columns_(required_columns) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open table: "+path);
        std::string line;
        while (std::getline(input,line)) {
            if (line.empty()||line[0]=='#') continue;
            std::istringstream row(line);
            std::vector<double> values(
                static_cast<std::size_t>(required_columns_),0.0);
            for (double& value:values) {
                if (!(row>>value)) {
                    throw std::runtime_error("invalid table row: "+path);
                }
            }
            const double k=values[0];
            const double value=values.at(
                static_cast<std::size_t>(target_column_));
            if (!(k>0.0)||!(value>0.0)) {
                throw std::runtime_error("table contains non-positive k/value");
            }
            if (!log_k_.empty()&&!(std::log(k)>log_k_.back())) {
                throw std::runtime_error("table k values are not increasing");
            }
            log_k_.push_back(std::log(k));
            log_value_.push_back(std::log(value));
        }
        if (log_k_.size()<8) throw std::runtime_error("table is too short");
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
        const double fraction=(x-log_k_[left])
                              /(log_k_[right]-log_k_[left]);
        return std::exp(
            log_value_[left]
            +fraction*(log_value_[right]-log_value_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("LogColumnTable has no Cosmology object");
    }

private:
    int target_column_;
    int required_columns_;
    std::vector<double> log_k_;
    std::vector<double> log_value_;
};

struct IntegerMode {
    int x=0;
    int y=0;
    int z=0;
    int shell=-1;
    double magnitude=0.0;
};

int lookup_offset(int x,int y,int z) {
    return (x+kMaximumIndex)*kLookupSide*kLookupSide
           +(y+kMaximumIndex)*kLookupSide
           +(z+kMaximumIndex);
}

double fft_component(int index) {
    /*
     * 精确复现 ``2*pi*np.fft.fftfreq(N,d=L/N)`` 的运算顺序。
     * 不能简写成 index*(2*pi/L)：在 7*kf、11*kf 边界上，最后一位舍入会
     * 随 Cartesian orientation 改变 NumPy shell assignment。
     */
    const double spacing=kBoxSize/static_cast<double>(kMeshSize);
    const double frequency=
        static_cast<double>(index)
        *(1.0/(static_cast<double>(kMeshSize)*spacing));
    return 2.0*kPi*frequency;
}

int shell_for_magnitude(double magnitude) {
    for (int shell=0;shell<kShellCount;++shell) {
        const bool upper=
            shell==kShellCount-1
            ?magnitude<=kEdges[static_cast<std::size_t>(shell+1)]
                 +64.0*std::numeric_limits<double>::epsilon()
            :magnitude<kEdges[static_cast<std::size_t>(shell+1)];
        if (magnitude>=kEdges[static_cast<std::size_t>(shell)]&&upper) {
            return shell;
        }
    }
    return -1;
}

std::vector<IntegerMode> selected_modes(std::vector<int>& lookup) {
    lookup.assign(
        static_cast<std::size_t>(kLookupSide*kLookupSide*kLookupSide),-1);
    std::vector<IntegerMode> result;
    for (int x=-kMaximumIndex;x<=kMaximumIndex;++x) {
        for (int y=-kMaximumIndex;y<=kMaximumIndex;++y) {
            for (int z=-kMaximumIndex;z<=kMaximumIndex;++z) {
                const double kx=fft_component(x);
                const double ky=fft_component(y);
                const double kz=fft_component(z);
                const double magnitude=std::sqrt(
                    kx*kx+ky*ky+kz*kz);
                const int shell=shell_for_magnitude(magnitude);
                if (shell<0) continue;
                const int index=static_cast<int>(result.size());
                result.push_back(IntegerMode{x,y,z,shell,magnitude});
                lookup.at(static_cast<std::size_t>(lookup_offset(x,y,z)))=
                    index;
            }
        }
    }
    return result;
}

struct TriangleLabel {
    int first=0;
    int second=0;
    int third=0;
};

constexpr int kRoleCount=8;
const std::array<const char*,kRoleCount> kRoleNames{{
    "mmm",
    "hmm_hleg0",
    "hmm_hleg1",
    "hmm_hleg2",
    "hhm_mleg0",
    "hhm_mleg1",
    "hhm_mleg2",
    "hhh"
}};
/* Bit ``leg`` is one when that ordered Fourier leg is a halo field. */
const std::array<int,kRoleCount> kRoleMasks{{0,1,2,4,6,5,3,7}};

/* 独立校准候选基底；仅导出，不自动加入已冻结的 deterministic total。
 * beta_k2 是 fNL*(-laplacian delta) 的单位系数，含 K1、K2 和位移 product rule。
 * N 为完整 halo 白噪声，Np=1/n_h；N2 形状的系数是 N*(N-Np)。
 * 噪声方向使用完整 halo 位移源和 exact signed-object factorial ledger。
 */
constexpr int kRepairBasisCount=17;
const std::array<const char*,kRepairBasisCount> kRepairBasisNames{{
    "beta_k2_f2","beta_k2_b2","beta_k2_bK2","beta_k2_reconstruction",
    "noise_N","noise_Np","noise_N2_minus_NNp",
    "noise_N_dK1Psi","noise_Np_dK1Psi","noise_N_dK1k2","noise_Np_dK1k2",
    "factorial_Cdelta","factorial_Cpsi","factorial_Ck2","factorial_T",
    "factorial_Cdelta_dK1Psi","factorial_Cdelta_dK1k2"
}};

struct RoleTreeResult {
    double gaussian=0.0;
    double response=0.0;
    // 二阶有限 fNL 项：两条线性 halo-Psi 腿同时插入 Gaussian K2。
    double quadratic_response=0.0;
    std::array<double,kRepairBasisCount> repair{};
};

std::vector<TriangleLabel> triangle_labels(
    std::array<std::array<std::array<int,kShellCount>,kShellCount>,kShellCount>& lookup) {
    for (auto& first:lookup) {
        for (auto& second:first) second.fill(-1);
    }
    std::vector<TriangleLabel> result;
    for (int first=0;first<kShellCount;++first) {
        for (int second=first;second<kShellCount;++second) {
            for (int third=second;third<kShellCount;++third) {
                const int index=static_cast<int>(result.size());
                result.push_back(TriangleLabel{first,second,third});
                lookup[static_cast<std::size_t>(first)]
                      [static_cast<std::size_t>(second)]
                      [static_cast<std::size_t>(third)]=index;
            }
        }
    }
    if (result.size()!=kCombinationCount) {
        throw std::logic_error("unexpected shell-combination count");
    }
    return result;
}

struct Accumulation {
    std::uint64_t count=0;
    long double k1_sum=0.0L;
    long double k2_sum=0.0L;
    long double k3_sum=0.0L;
    long double Btree=0.0L;
    long double total=0.0L;
    long double primordial=0.0L;
    long double bPsi_f2=0.0L;
    long double bPsi_advection=0.0L;
    long double bPsiDelta=0.0L;
    long double bPsi_b2=0.0L;
    long double bPsi_bK2=0.0L;
    long double bPsiS=0.0L;
    long double bPsi_reconstruction=0.0L;
    long double stochastic_PsiP=0.0L;
    long double stochastic_P=0.0L;
    long double stochastic_constant=0.0L;
    std::array<long double,kRoleCount> role_gaussian{};
    std::array<long double,kRoleCount> role_response{};
    std::array<long double,kRoleCount> role_quadratic_response{};
    std::array<std::array<long double,kRepairBasisCount>,kRoleCount> repair_basis{};

    void add(
        const IntegerMode& first,
        const IntegerMode& second,
        const IntegerMode& third,
        const native::ComponentResult& value,
        const std::array<RoleTreeResult,kRoleCount>& roles) {
        ++count;
        k1_sum+=first.magnitude;
        k2_sum+=second.magnitude;
        k3_sum+=third.magnitude;
        Btree+=value.Btree;
        total+=value.dBdfNL_halo_png_tree;
        primordial+=value.dBdfNL_halo_png_primordial;
        bPsi_f2+=value.dBdfNL_halo_bPsi_f2;
        bPsi_advection+=value.dBdfNL_halo_bPsi_advection;
        bPsiDelta+=value.dBdfNL_halo_bPsiDelta;
        bPsi_b2+=value.dBdfNL_halo_bPsi_b2;
        bPsi_bK2+=value.dBdfNL_halo_bPsi_bK2;
        bPsiS+=value.dBdfNL_halo_bPsiS;
        bPsi_reconstruction+=value.dBdfNL_halo_bPsi_reconstruction;
        stochastic_PsiP+=value.dBdfNL_halo_stochastic_PsiP_basis;
        stochastic_P+=value.dBdfNL_halo_stochastic_P_basis;
        stochastic_constant+=
            value.dBdfNL_halo_stochastic_constant_basis;
        for (int role=0;role<kRoleCount;++role) {
            role_gaussian[static_cast<std::size_t>(role)]+=
                roles[static_cast<std::size_t>(role)].gaussian;
            role_response[static_cast<std::size_t>(role)]+= 
                roles[static_cast<std::size_t>(role)].response;
            role_quadratic_response[static_cast<std::size_t>(role)]+=
                roles[static_cast<std::size_t>(role)].quadratic_response;
            for (int component=0;component<kRepairBasisCount;++component) {
                repair_basis[role][component]+=roles[role].repair[component];
            }
        }
    }

    void merge(const Accumulation& other) {
        count+=other.count;
        k1_sum+=other.k1_sum;
        k2_sum+=other.k2_sum;
        k3_sum+=other.k3_sum;
        Btree+=other.Btree;
        total+=other.total;
        primordial+=other.primordial;
        bPsi_f2+=other.bPsi_f2;
        bPsi_advection+=other.bPsi_advection;
        bPsiDelta+=other.bPsiDelta;
        bPsi_b2+=other.bPsi_b2;
        bPsi_bK2+=other.bPsi_bK2;
        bPsiS+=other.bPsiS;
        bPsi_reconstruction+=other.bPsi_reconstruction;
        stochastic_PsiP+=other.stochastic_PsiP;
        stochastic_P+=other.stochastic_P;
        stochastic_constant+=other.stochastic_constant;
        for (int role=0;role<kRoleCount;++role) {
            role_gaussian[static_cast<std::size_t>(role)]+=
                other.role_gaussian[static_cast<std::size_t>(role)];
            role_response[static_cast<std::size_t>(role)]+=
                other.role_response[static_cast<std::size_t>(role)];
            role_quadratic_response[static_cast<std::size_t>(role)]+=
                other.role_quadratic_response[static_cast<std::size_t>(role)];
            for (int component=0;component<kRepairBasisCount;++component) {
                repair_basis[role][component]+=other.repair_basis[role][component];
            }
        }
    }
};

double mean(long double sum,std::uint64_t count) {
    return static_cast<double>(sum/static_cast<long double>(count));
}

native::ClosedTriangleVectors closed_vectors(
    const IntegerMode& first,
    const IntegerMode& second,
    const IntegerMode& third) {
    return native::ClosedTriangleVectors{{
        {{fft_component(first.x),fft_component(first.y),fft_component(first.z)}},
        {{fft_component(second.x),fft_component(second.y),fft_component(second.z)}},
        {{fft_component(third.x),fft_component(third.y),fft_component(third.z)}}}};
}

double dot3(
    const std::array<real,3>& first,
    const std::array<real,3>& second) {
    return static_cast<double>(first[0])*static_cast<double>(second[0])
          +static_cast<double>(first[1])*static_cast<double>(second[1])
          +static_cast<double>(first[2])*static_cast<double>(second[2]);
}

double norm3(const std::array<real,3>& vector) {
    return std::sqrt(dot3(vector,vector));
}

std::array<real,3> add3(
    const std::array<real,3>& first,
    const std::array<real,3>& second) {
    return {{
        static_cast<real>(first[0]+second[0]),
        static_cast<real>(first[1]+second[1]),
        static_cast<real>(first[2]+second[2])}};
}

double sinc_unscaled(double value) {
    return std::abs(value)<1.0e-14 ? 1.0 : std::sin(value)/value;
}

double role_reconstruction_window(
    const std::array<real,3>& vector,
    const native::NativeConfig& config) {
    const double magnitude=norm3(vector);
    double window=std::exp(
        -0.5*std::pow(magnitude*config.smoothing_radius,2));
    if (config.recon_cellsize>0.0&&config.recon_cic_window_power>0) {
        double cartesian=1.0;
        for (const real component:vector) {
            cartesian*=sinc_unscaled(
                0.5*static_cast<double>(component)*config.recon_cellsize);
        }
        window*=std::pow(cartesian,config.recon_cic_window_power);
    }
    return window;
}

double role_shift_factor(
    const std::array<real,3>& total,
    const std::array<real,3>& input,
    const native::NativeConfig& config) {
    return -role_reconstruction_window(input,config)/config.bias_recon
           *dot3(total,input)/dot3(input,input);
}

double f2_kernel(double first,double second,double mu) {
    return 5.0/7.0+0.5*mu*(first/second+second/first)
           +2.0/7.0*mu*mu;
}

double equilateral_b111(
    const std::array<double,3>& linear_power,
    const std::array<double,3>& transfer) {
    std::array<double,3> potential{};
    for (int leg=0;leg<3;++leg) {
        potential[static_cast<std::size_t>(leg)]=
            linear_power[static_cast<std::size_t>(leg)]
            /std::pow(transfer[static_cast<std::size_t>(leg)],2);
    }
    const double pair_sum=
        potential[0]*potential[1]+potential[1]*potential[2]
        +potential[2]*potential[0];
    const double product_term=std::pow(
        potential[0]*potential[1]*potential[2],2.0/3.0);
    double permutations=0.0;
    for (int first=0;first<3;++first) {
        for (int second=0;second<3;++second) {
            if (first==second) continue;
            const int third=3-first-second;
            if (third<0||third>2||third==first||third==second) continue;
            permutations+=std::pow(
                potential[static_cast<std::size_t>(first)],1.0/3.0)
                *std::pow(
                    potential[static_cast<std::size_t>(second)],2.0/3.0)
                *potential[static_cast<std::size_t>(third)];
        }
    }
    return transfer[0]*transfer[1]*transfer[2]
           *6.0*(-pair_sum-2.0*product_term+permutations);
}

RoleTreeResult compute_role_tree(
    const native::ClosedTriangleVectors& vectors,
    const PowerSpectrum& power,
    const PowerSpectrum& transfer,
    const native::NativeConfig& config,
    const native::HaloDelta2PNGBiasParams& bias,
    bool post,
    int halo_mask) {
    std::array<double,3> magnitude{};
    std::array<double,3> linear_power{};
    std::array<double,3> transfer_value{};
    std::array<double,3> psi_shape{};
    for (int leg=0;leg<3;++leg) {
        magnitude[static_cast<std::size_t>(leg)]=
            norm3(vectors[static_cast<std::size_t>(leg)]);
        linear_power[static_cast<std::size_t>(leg)]=
            power.Evaluate(magnitude[static_cast<std::size_t>(leg)]);
        transfer_value[static_cast<std::size_t>(leg)]=
            transfer.Evaluate(magnitude[static_cast<std::size_t>(leg)]);
        psi_shape[static_cast<std::size_t>(leg)]=
            std::pow(magnitude[static_cast<std::size_t>(leg)]/bias.k_pivot,2)
            /transfer_value[static_cast<std::size_t>(leg)];
    }
    const auto is_halo=[halo_mask](int leg) {
        return (halo_mask&(1<<leg))!=0;
    };
    const auto k1_gaussian=[&bias,&is_halo](int leg) {
        return is_halo(leg)?static_cast<double>(bias.b1):1.0;
    };
    const auto k1_png=[&bias,&is_halo,&psi_shape](int leg) {
        return is_halo(leg)
            ?static_cast<double>(bias.bPsi)*psi_shape[static_cast<std::size_t>(leg)]
            :0.0;
    };
    RoleTreeResult result;
    // T 是经过 exact factorial contact subtraction 后的三点常数，只出现在 hhh。
    result.repair[14]=(halo_mask==7)?1.0:0.0;
    result.response=equilateral_b111(linear_power,transfer_value)
                    *k1_gaussian(0)*k1_gaussian(1)*k1_gaussian(2);
    const std::array<std::array<int,3>,3> cyclic{{
        {{0,1,2}},{{1,2,0}},{{2,0,1}}}};
    for (const auto& row:cyclic) {
        const int first=row[0];
        const int second=row[1];
        const int output=row[2];
        const double mu=dot3(
            vectors[static_cast<std::size_t>(first)],
            vectors[static_cast<std::size_t>(second)])
            /(magnitude[static_cast<std::size_t>(first)]
              *magnitude[static_cast<std::size_t>(second)]);
        const double f2=f2_kernel(
            magnitude[static_cast<std::size_t>(first)],
            magnitude[static_cast<std::size_t>(second)],mu);
        double k2_gaussian=f2;
        double k2_png=0.0;
        if (is_halo(output)) {
            k2_gaussian=bias.b1*k2_gaussian+0.5*bias.b2
                        +bias.bK2*(mu*mu-1.0/3.0);
            const double s_first=psi_shape[static_cast<std::size_t>(first)];
            const double s_second=psi_shape[static_cast<std::size_t>(second)];
            k2_png=
                0.5*bias.bPsi*mu*(
                    magnitude[static_cast<std::size_t>(first)]
                    /magnitude[static_cast<std::size_t>(second)]*s_first
                    +magnitude[static_cast<std::size_t>(second)]
                    /magnitude[static_cast<std::size_t>(first)]*s_second)
                +0.5*bias.bPsiDelta*(s_first+s_second)
                +0.25*bias.bPsiS*(3.0*mu*mu-1.0)
                    *(s_first+s_second);
        }
        const double first_g=k1_gaussian(first);
        const double second_g=k1_gaussian(second);
        const double output_g=k1_gaussian(output);
        const double ksq_first=magnitude[first]*magnitude[first];
        const double ksq_second=magnitude[second]*magnitude[second];
        const double ksq_output=magnitude[output]*magnitude[output];
        const double direction_first=is_halo(first)?ksq_first:0.0;
        const double direction_second=is_halo(second)?ksq_second:0.0;
        const double direction_output=is_halo(output)?ksq_output:0.0;
        const double pair_power=linear_power[first]*linear_power[second];
        const double external_direction=
            direction_first*second_g+first_g*direction_second;
        // 固定的残余随机算符：两条 halo 噪声腿，剩余腿携带 Gaussian/Psi/k^2 场。
        // Cdelta= <epsilon epsilon_delta> - Np*b1/2；Cpsi/Ck2 使用同样的
        // factorial renormalization 条件。它们在 pre Bhmm 中严格消失。
        const double pair_is_halo=(is_halo(first)&&is_halo(second))?1.0:0.0;
        const double stochastic_delta=2.0*pair_is_halo*output_g*linear_power[output];
        result.repair[11]+=stochastic_delta;
        result.repair[12]+=stochastic_delta*psi_shape[output];
        result.repair[13]+=stochastic_delta*ksq_output;
        result.repair[15]+=2.0*pair_is_halo*(is_halo(output)?1.0:0.0)
                           *psi_shape[output]*linear_power[output];
        result.repair[16]+=2.0*pair_is_halo*(is_halo(output)?1.0:0.0)
                           *ksq_output*linear_power[output];
        result.repair[0]+=2.0*pair_power*f2*(
            external_direction*output_g+first_g*second_g*direction_output);
        if (is_halo(output)) {
            result.repair[1]+=pair_power*external_direction*bias.b2;
            result.repair[2]+=2.0*pair_power*external_direction*bias.bK2*(mu*mu-1.0/3.0);
        }
        if (post) {
            const std::array<real,3> total=add3(
                vectors[static_cast<std::size_t>(first)],
                vectors[static_cast<std::size_t>(second)]);
            const double shift_first=role_shift_factor(
                total,vectors[static_cast<std::size_t>(first)],config);
            const double shift_second=role_shift_factor(
                total,vectors[static_cast<std::size_t>(second)],config);
            const double target_g=is_halo(output)?bias.b1:1.0;
            const double target_p_first=is_halo(output)
                ?bias.bPsi*psi_shape[static_cast<std::size_t>(first)]:0.0;
            const double target_p_second=is_halo(output)
                ?bias.bPsi*psi_shape[static_cast<std::size_t>(second)]:0.0;
            const double source_g=bias.b1;
            const double source_p_first=
                bias.bPsi*psi_shape[static_cast<std::size_t>(first)];
            const double source_p_second=
                bias.bPsi*psi_shape[static_cast<std::size_t>(second)];
            k2_gaussian+=0.5*source_g*target_g
                         *(shift_first+shift_second);
            k2_png+=0.5*(
                (source_p_first*target_g+source_g*target_p_second)
                    *shift_first
                +(source_p_second*target_g+source_g*target_p_first)
                    *shift_second);
            const double rec_g=0.5*source_g*target_g*(shift_first+shift_second);
            const double rec_k2_direction=0.5*(
                (ksq_first*target_g+source_g*(is_halo(output)?ksq_second:0.0))*shift_first
                +(ksq_second*target_g+source_g*(is_halo(output)?ksq_first:0.0))*shift_second);
            result.repair[3]+=2.0*pair_power*(
                external_direction*rec_g+first_g*second_g*rec_k2_direction);

            // 逐 Wick 配对导出 N 系数及 K1=Psi/k^2 的独立方向；random 不入位移源。
            for (int order=0;order<2;++order) {
                const int a=order==0?first:second;
                const int b=order==0?second:first;
                const double sf=order==0?shift_first:shift_second;
                const double ia=is_halo(a)?1.0:0.0;
                const double ib=is_halo(b)?1.0:0.0;
                const double ic=is_halo(output)?1.0:0.0;
                const double qa=k1_gaussian(a),qb=k1_gaussian(b),qc=output_g;
                result.repair[4]+=sf*(ia*qb*qc*linear_power[b]
                                          +qa*bias.b1*linear_power[a]*ib*ic);
                for (int direction=0;direction<2;++direction) {
                    const double da=direction==0?psi_shape[a]:magnitude[a]*magnitude[a];
                    const double db=direction==0?psi_shape[b]:magnitude[b]*magnitude[b];
                    result.repair[7+2*direction]+=sf*(
                        ia*(ib*db*qc+qb*ic*db)*linear_power[b]
                        +(ia*da*bias.b1+qa*da)*linear_power[a]*ib*ic);
                }
            }
            // D2_{ij} 与剩余场的 contact 扣项，符号保留在模板内。
            const double u=role_shift_factor(vectors[output],vectors[output],config);
            const double contact=(is_halo(first)&&is_halo(second))?1.0:0.0;
            const double ih=is_halo(output)?1.0:0.0;
            result.repair[5]-=contact*u*output_g*bias.b1*linear_power[output];
            result.repair[6]+=contact*ih*u;
            result.repair[8]-=contact*u*(ih*bias.b1+output_g)*psi_shape[output]*linear_power[output];
            result.repair[10]-=contact*u*(ih*bias.b1+output_g)*ksq_output*linear_power[output];
        }
        const double first_p=k1_png(first);
        const double second_p=k1_png(second);
        result.gaussian+=2.0*first_g*second_g*k2_gaussian*pair_power;
        result.response+=2.0*pair_power*(
            first_g*second_g*k2_png
            +(first_p*second_g+first_g*second_p)*k2_gaussian);
        // 有限 fNL^2 的 tree 项，供 EQ_p/EQ_m 对称分量使用。
        result.quadratic_response+=2.0*pair_power*first_p*second_p*k2_gaussian;
    }
    return result;
}

std::array<RoleTreeResult,kRoleCount> compute_all_roles(
    const native::ClosedTriangleVectors& vectors,
    const PowerSpectrum& power,
    const PowerSpectrum& transfer,
    const native::NativeConfig& config,
    const native::HaloDelta2PNGBiasParams& bias,
    bool post) {
    std::array<RoleTreeResult,kRoleCount> result{};
    for (int role=0;role<kRoleCount;++role) {
        result[static_cast<std::size_t>(role)]=compute_role_tree(
            vectors,power,transfer,config,bias,post,
            kRoleMasks[static_cast<std::size_t>(role)]);
    }
    return result;
}

void write_string(const std::string& value) {
    std::cout<<'"';
    for (const char character:value) {
        if (character=='"'||character=='\\') std::cout<<'\\';
        std::cout<<character;
    }
    std::cout<<'"';
}

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=15) {
            std::cerr
                <<"usage: marisa_b_eq_strict_tree_projector "
                <<"POWER_TABLE PNG_TABLE pre|post B1 B2 BK2 "
                <<"BPSI BPSIDELTA BPSIS KPIVOT R_SMOOTH B_REC "
                <<"CELL_SIZE CIC_POWER\n";
            return 2;
        }
        const std::string power_path=argv[1];
        const std::string png_path=argv[2];
        const std::string mode=argv[3];
        if (mode!="pre"&&mode!="post") {
            throw std::invalid_argument("mode must be pre or post");
        }
        native::HaloDelta2PNGBiasParams bias;
        bias.b1=parse_double(argv[4],"b1");
        bias.b2=parse_double(argv[5],"b2");
        bias.bK2=parse_double(argv[6],"bK2");
        bias.bPsi=parse_double(argv[7],"bPsi");
        bias.bPsiDelta=parse_double(argv[8],"bPsiDelta");
        bias.bPsiS=parse_double(argv[9],"bPsiS");
        bias.k_pivot=parse_double(argv[10],"k_pivot");
        native::NativeConfig config;
        config.smoothing_radius=parse_double(argv[11],"R_smooth");
        config.bias_recon=parse_double(argv[12],"b_rec");
        config.recon_cellsize=parse_double(argv[13],"cell_size");
        config.recon_cic_window_power=parse_int(argv[14],"CIC power");
        config.singular_floor=1.0e-12;
        if (!(bias.k_pivot>0.0)||!(config.smoothing_radius>0.0)
            ||!(config.bias_recon>0.0)||config.recon_cellsize<0.0
            ||config.recon_cic_window_power<0) {
            throw std::invalid_argument("invalid bias/reconstruction configuration");
        }

        const LogColumnTable power(power_path,1,4);
        const LogColumnTable transfer(png_path,3,4);
        std::vector<int> mode_lookup;
        const std::vector<IntegerMode> modes=selected_modes(mode_lookup);
        std::array<std::uint64_t,kShellCount> shell_mode_counts{};
        for (const IntegerMode& item:modes) {
            ++shell_mode_counts[static_cast<std::size_t>(item.shell)];
        }
        std::array<std::array<std::array<int,kShellCount>,kShellCount>,kShellCount>
            triangle_lookup{};
        const std::vector<TriangleLabel> labels=
            triangle_labels(triangle_lookup);

        const int requested_threads=std::max(1,omp_get_max_threads());
        const int threads=std::min(8,requested_threads);
        omp_set_num_threads(threads);
        std::vector<std::vector<Accumulation>> local(
            static_cast<std::size_t>(threads),
            std::vector<Accumulation>(kCombinationCount));

#pragma omp parallel
        {
            const int thread=omp_get_thread_num();
            const std::size_t begin=
                modes.size()*static_cast<std::size_t>(thread)
                /static_cast<std::size_t>(threads);
            const std::size_t end=
                modes.size()*static_cast<std::size_t>(thread+1)
                /static_cast<std::size_t>(threads);
            std::vector<Accumulation>& accumulator=
                local.at(static_cast<std::size_t>(thread));
            for (std::size_t first_index=begin;first_index<end;++first_index) {
                const IntegerMode& first=modes[first_index];
                for (const IntegerMode& second:modes) {
                    const int cx=-first.x-second.x;
                    const int cy=-first.y-second.y;
                    const int cz=-first.z-second.z;
                    if (cx<-kMaximumIndex||cx>kMaximumIndex
                        ||cy<-kMaximumIndex||cy>kMaximumIndex
                        ||cz<-kMaximumIndex||cz>kMaximumIndex) {
                        continue;
                    }
                    const int third_index=mode_lookup.at(
                        static_cast<std::size_t>(lookup_offset(cx,cy,cz)));
                    if (third_index<0) continue;
                    const IntegerMode& third=modes.at(
                        static_cast<std::size_t>(third_index));
                    if (!(first.shell<=second.shell
                          &&second.shell<=third.shell)) {
                        continue;
                    }
                    const int triangle_index=
                        triangle_lookup[static_cast<std::size_t>(first.shell)]
                                       [static_cast<std::size_t>(second.shell)]
                                       [static_cast<std::size_t>(third.shell)];
                    if (triangle_index<0) {
                        throw std::logic_error("missing triangle label");
                    }
                    const native::ClosedTriangleVectors vectors=
                        closed_vectors(first,second,third);
                    const native::ComponentResult value=mode=="pre"
                        ?native::compute_pre_recon_halo_delta2_png_tree_dfNL_vectors(
                             native::PrimordialBispectrumShape::Equilateral,
                             power,transfer,vectors,bias)
                        :native::compute_post_recon_halo_delta2_png_tree_dfNL_vectors(
                             native::PrimordialBispectrumShape::Equilateral,
                             power,transfer,vectors,config,bias);
                    const std::array<RoleTreeResult,kRoleCount> roles=
                        compute_all_roles(
                            vectors,power,transfer,config,bias,mode=="post");
                    accumulator.at(
                        static_cast<std::size_t>(triangle_index))
                        .add(first,second,third,value,roles);
                }
            }
        }

        std::vector<Accumulation> merged(kCombinationCount);
        for (int thread=0;thread<threads;++thread) {
            for (int triangle=0;triangle<kCombinationCount;++triangle) {
                merged[static_cast<std::size_t>(triangle)].merge(
                    local[static_cast<std::size_t>(thread)]
                         [static_cast<std::size_t>(triangle)]);
            }
        }
        std::uint64_t total_count=0;
        int nonempty=0;
        for (const Accumulation& value:merged) {
            total_count+=value.count;
            nonempty+=value.count>0;
        }

        std::cout<<std::setprecision(17);
        std::cout<<"{\n";
        std::cout<<"  \"schema\": \"marisa-b-eq-strict-delta2-tree-projector-v1\",\n";
        std::cout<<"  \"mode\": ";write_string(mode);std::cout<<",\n";
        std::cout<<"  \"shape\": \"equilateral\",\n";
        std::cout<<"  \"boxsize_mpc_h\": "<<kBoxSize<<",\n";
        std::cout<<"  \"meshsize\": "<<kMeshSize<<",\n";
        std::cout<<"  \"all_legs_kmax_h_mpc\": "<<kMaximum<<",\n";
        std::cout<<"  \"selected_mode_count\": "<<modes.size()<<",\n";
        std::cout<<"  \"shell_mode_counts\": [";
        for (std::size_t shell=0;shell<shell_mode_counts.size();++shell) {
            if (shell) std::cout<<',';
            std::cout<<shell_mode_counts[shell];
        }
        std::cout<<"],\n";
        std::cout<<"  \"fixed_shell_order_triplet_count\": "<<total_count<<",\n";
        std::cout<<"  \"nonempty_triangle_count\": "<<nonempty<<",\n";
        std::cout<<"  \"threads_used\": "<<threads<<",\n";
        std::cout<<"  \"power_table\": ";write_string(power_path);std::cout<<",\n";
        std::cout<<"  \"png_table\": ";write_string(png_path);std::cout<<",\n";
        std::cout<<"  \"bias\": {\"b1\": "<<bias.b1
                 <<", \"b2\": "<<bias.b2
                 <<", \"bK2\": "<<bias.bK2
                 <<", \"bPsi\": "<<bias.bPsi
                 <<", \"bPsiDelta\": "<<bias.bPsiDelta
                 <<", \"bPsiS\": "<<bias.bPsiS
                 <<", \"k_pivot_h_mpc\": "<<bias.k_pivot<<"},\n";
        std::cout<<"  \"reconstruction\": {\"R_mpc_h\": "
                 <<config.smoothing_radius
                 <<", \"b_rec\": "<<config.bias_recon
                 <<", \"cellsize_mpc_h\": "<<config.recon_cellsize
                 <<", \"cic_window_power\": "
                 <<config.recon_cic_window_power<<"},\n";
        std::cout<<"  \"empirical_TG\": false,\n";
        std::cout<<"  \"response_gain\": false,\n";
        std::cout<<"  \"results\": [\n";
        bool first_output=true;
        for (std::size_t index=0;index<merged.size();++index) {
            const Accumulation& value=merged[index];
            if (value.count==0) continue;
            if (!first_output) std::cout<<",\n";
            first_output=false;
            const TriangleLabel& label=labels[index];
            std::cout<<"    {\"shells\": ["<<label.first<<','
                     <<label.second<<','<<label.third<<"], "
                     <<"\"count\": "<<value.count<<", "
                     <<"\"k_mean_h_mpc\": ["
                     <<mean(value.k1_sum,value.count)<<','
                     <<mean(value.k2_sum,value.count)<<','
                     <<mean(value.k3_sum,value.count)<<"], "
                     <<"\"Btree\": "<<mean(value.Btree,value.count)<<", "
                     <<"\"dBdfNL_halo_png_tree\": "<<mean(value.total,value.count)<<", "
                     <<"\"dBdfNL_halo_png_primordial\": "<<mean(value.primordial,value.count)<<", "
                     <<"\"dBdfNL_halo_bPsi_f2\": "<<mean(value.bPsi_f2,value.count)<<", "
                     <<"\"dBdfNL_halo_bPsi_advection\": "<<mean(value.bPsi_advection,value.count)<<", "
                     <<"\"dBdfNL_halo_bPsiDelta\": "<<mean(value.bPsiDelta,value.count)<<", "
                     <<"\"dBdfNL_halo_bPsi_b2\": "<<mean(value.bPsi_b2,value.count)<<", "
                     <<"\"dBdfNL_halo_bPsi_bK2\": "<<mean(value.bPsi_bK2,value.count)<<", "
                     <<"\"dBdfNL_halo_bPsiS\": "<<mean(value.bPsiS,value.count)<<", "
                     <<"\"dBdfNL_halo_bPsi_reconstruction\": "<<mean(value.bPsi_reconstruction,value.count)<<", "
                     <<"\"dBdfNL_halo_stochastic_PsiP_basis\": "<<mean(value.stochastic_PsiP,value.count)<<", "
                     <<"\"dBdfNL_halo_stochastic_P_basis\": "<<mean(value.stochastic_P,value.count)<<", "
                     <<"\"dBdfNL_halo_stochastic_constant_basis\": "<<mean(value.stochastic_constant,value.count)
                     ;
            for (int role=0;role<kRoleCount;++role) {
                std::cout<<", \"gaussian_tree_"
                         <<kRoleNames[static_cast<std::size_t>(role)]<<"\": "
                         <<mean(
                             value.role_gaussian[static_cast<std::size_t>(role)],
                             value.count)
                         <<", \"dBdfNL_tree_"
                         <<kRoleNames[static_cast<std::size_t>(role)]<<"\": "
                         <<mean(
                             value.role_response[static_cast<std::size_t>(role)],
                             value.count);
                std::cout<<", \"quadratic_dBdfNL2_tree_"
                         <<kRoleNames[static_cast<std::size_t>(role)]<<"\": "
                         <<mean(value.role_quadratic_response[static_cast<std::size_t>(role)], value.count);
                for (int component=0;component<kRepairBasisCount;++component) {
                    std::cout<<", \"repair_"<<kRepairBasisNames[component]
                             <<'_'<<kRoleNames[role]<<"\": "
                             <<mean(value.repair_basis[role][component],value.count);
                }
            }
            std::cout<<'}';
        }
        std::cout<<"\n  ]\n}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"MARISA-B EQ strict tree projector failed: "
                 <<error.what()<<'\n';
        return 1;
    }
}

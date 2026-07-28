#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PowerSpectrum.h"
#include "field_kernel_provider.h"
#include "lattice_shell_rule.h"
#include "marisa_b_native.h"
#include "parameter_registry.h"
#include "poisson_reconstruction.h"
#include "stochastic.h"

namespace eft=marisa_b_eft_v2;
namespace hv1=marisa_b_halo_v1;
namespace native=marisa_b_native;
namespace shell=marisa_b_shell_v1;

namespace {

std::string file_sha256(const std::string& path) {
    std::ifstream input(path,std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "cannot hash provenance input: "+path);
    }
    const std::string payload{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    return eft::sha256_hex(payload);
}

class TabulatedPower final:public PowerSpectrum {
public:
    explicit TabulatedPower(const std::string& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error(
                "cannot open linear-power table: "+path);
        }
        std::string line;
        while (std::getline(input,line)) {
            if (line.empty() || line[0]=='#') continue;
            std::istringstream row(line);
            double k=0.0;
            double power=0.0;
            if (!(row>>k>>power)
                ||!(k>0.0 && power>0.0)) {
                throw std::runtime_error(
                    "invalid positive linear-power table row");
            }
            log_k_.push_back(std::log(k));
            log_power_.push_back(std::log(power));
        }
        if (log_k_.size()<2) {
            throw std::runtime_error(
                "linear-power table is too short");
        }
    }

    real Evaluate(real k) const override {
        if (!(k>0.0)) return 0.0;
        const double x=std::log(k);
        std::size_t right=1;
        if (x>=log_k_.back()) {
            right=log_k_.size()-1;
        } else if (x>log_k_.front()) {
            right=static_cast<std::size_t>(
                std::upper_bound(
                    log_k_.begin(),log_k_.end(),x)
                -log_k_.begin());
        }
        const std::size_t left=right-1;
        const double fraction=
            (x-log_k_[left])
            /(log_k_[right]-log_k_[left]);
        return std::exp(
            log_power_[left]
            +fraction*(log_power_[right]
                       -log_power_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error(
            "TabulatedPower has no Cosmology object");
    }

private:
    std::vector<double> log_k_;
    std::vector<double> log_power_;
};

class TabulatedColumn final:public PowerSpectrum {
public:
    TabulatedColumn(
        const std::string& path,
        int column):column_(column) {
        if (column_!=1 && column_!=3) {
            throw std::invalid_argument(
                "tabulated column must be P_L=1 or M=3");
        }
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error(
                "cannot open local-PNG table: "+path);
        }
        std::string line;
        while (std::getline(input,line)) {
            if (line.empty() || line[0]=='#') continue;
            std::istringstream row(line);
            std::array<double,4> values{};
            if (!(row>>values[0]>>values[1]
                     >>values[2]>>values[3])
                ||!(values[0]>0.0)
                ||!(values[static_cast<std::size_t>(
                        column_)]>0.0)) {
                throw std::runtime_error(
                    "invalid positive local-PNG table row");
            }
            log_k_.push_back(std::log(values[0]));
            log_value_.push_back(std::log(
                values[static_cast<std::size_t>(column_)]));
        }
        if (log_k_.size()<2) {
            throw std::runtime_error(
                "local-PNG table is too short");
        }
    }

    real Evaluate(real k) const override {
        if (!(k>0.0)) return 0.0;
        const double x=std::log(k);
        std::size_t right=1;
        if (x>=log_k_.back()) {
            right=log_k_.size()-1;
        } else if (x>log_k_.front()) {
            right=static_cast<std::size_t>(
                std::upper_bound(
                    log_k_.begin(),log_k_.end(),x)
                -log_k_.begin());
        }
        const std::size_t left=right-1;
        const double fraction=
            (x-log_k_[left])
            /(log_k_[right]-log_k_[left]);
        return std::exp(
            log_value_[left]
            +fraction*(log_value_[right]
                       -log_value_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error(
            "TabulatedColumn has no Cosmology object");
    }

private:
    int column_;
    std::vector<double> log_k_;
    std::vector<double> log_value_;
};

class LocalPngK1Provider final:public eft::FieldKernelProvider {
public:
    LocalPngK1Provider(
        const eft::FieldKernelProvider& base,
        const PowerSpectrum& transfer,
        double amplitude):
        base_(base),transfer_(transfer),amplitude_(amplitude) {}

    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        eft::SparsePolynomial result=
            base_.deterministic(momenta);
        if (momenta.size()==1) {
            const double k=hv1::norm(momenta.front());
            const double transfer=transfer_(k);
            if (transfer>0.0) {
                result+=eft::SparsePolynomial::constant(
                    amplitude_/transfer);
            }
        }
        return result;
    }

    std::string_view name() const noexcept override {
        return "eft_v2_local_png_K1_finite";
    }

private:
    const eft::FieldKernelProvider& base_;
    const PowerSpectrum& transfer_;
    double amplitude_;
};

int parse_int(
    const char* text,
    const std::string& name,
    int minimum=1) {
    std::size_t consumed=0;
    const int value=std::stoi(text,&consumed);
    if (consumed!=std::string(text).size()
        ||value<minimum) {
        throw std::invalid_argument(
            "invalid integer for "+name);
    }
    return value;
}

double parse_double(
    const char* text,
    const std::string& name) {
    std::size_t consumed=0;
    const double value=std::stod(text,&consumed);
    if (consumed!=std::string(text).size()
        ||!std::isfinite(value)) {
        throw std::invalid_argument(
            "invalid number for "+name);
    }
    return value;
}

std::vector<double> read_edges(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "cannot open radial edge file: "+path);
    }
    std::vector<double> result;
    std::string line;
    while (std::getline(input,line)) {
        if (line.empty() || line[0]=='#') continue;
        std::istringstream row(line);
        double value=0.0;
        if (!(row>>value)) {
            throw std::runtime_error(
                "invalid radial edge row");
        }
        result.push_back(value);
    }
    if (result.size()!=16) {
        throw std::runtime_error(
            "post R1 radial edge file must contain 16 values");
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
                    std::cout<<"\\u"<<std::hex
                             <<std::setw(4)
                             <<std::setfill('0')
                             <<static_cast<int>(character);
                    std::cout.flags(flags);
                    std::cout.fill(fill);
                } else {
                    std::cout<<character;
                }
        }
    }
    std::cout<<'"';
}

void write_polynomial(
    const eft::SparsePolynomial& polynomial) {
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

void write_finite_polynomial_coefficients(
    const eft::SparsePolynomial& gaussian,
    const eft::SparsePolynomial& linear,
    const eft::SparsePolynomial& quadratic) {
    std::cout<<"{\"gaussian\":";
    write_polynomial(gaussian);
    std::cout<<",\"linear_unit_bphi\":";
    write_polynomial(linear);
    std::cout<<",\"quadratic_unit_bphi2\":";
    write_polynomial(quadratic);
    std::cout<<'}';
}

struct PolynomialComparison {
    double maximum_absolute_error=0.0;
    double maximum_scale=0.0;

    double relative_error() const {
        return maximum_absolute_error
               /std::max(maximum_scale,1.0e-300);
    }
};

PolynomialComparison compare_polynomials(
    const eft::SparsePolynomial& actual,
    const eft::SparsePolynomial& expected) {
    PolynomialComparison result;
    const eft::SparsePolynomial difference=actual-expected;
    for (const auto& term:difference.terms()) {
        result.maximum_absolute_error=std::max(
            result.maximum_absolute_error,
            std::fabs(term.second));
    }
    for (const auto& term:actual.terms()) {
        result.maximum_scale=std::max(
            result.maximum_scale,
            std::fabs(term.second));
    }
    for (const auto& term:expected.terms()) {
        result.maximum_scale=std::max(
            result.maximum_scale,
            std::fabs(term.second));
    }
    return result;
}

void require_polynomial_close(
    const eft::SparsePolynomial& actual,
    const eft::SparsePolynomial& expected,
    const std::string& label) {
    const PolynomialComparison comparison=
        compare_polynomials(actual,expected);
    if (comparison.maximum_absolute_error>2.0e-9
        &&comparison.relative_error()>2.0e-12) {
        std::ostringstream message;
        message<<std::setprecision(17)
               <<label
               <<" failed: max_abs="
               <<comparison.maximum_absolute_error
               <<", max_rel="
               <<comparison.relative_error();
        throw std::runtime_error(message.str());
    }
}

native::ClosedTriangleVectors native_vectors(
    const std::array<hv1::Vec3,3>& vectors) {
    native::ClosedTriangleVectors result{};
    for (std::size_t index=0;index<3;++index) {
        result[index]={
            vectors[index].x,
            vectors[index].y,
            vectors[index].z};
    }
    return result;
}

struct NativeAccumulator {
    double stochastic_alpha3_basis=0.0;
    double dBdfNL_local_tree=0.0;
    double dBdfNL_local_primordial=0.0;
    double dBdfNL_local_bphi_f2=0.0;
    double dBdfNL_local_bphi_advection=0.0;
    double dBdfNL_local_bphidelta=0.0;
    double dBdfNL_local_bphi_b2=0.0;
    double dBdfNL_local_bphi_bK2=0.0;
    double dBdfNL_local_bphi_reconstruction=0.0;
    double dBdfNL_stochastic_alpha3_basis=0.0;
    double Bhalo_tree_fNL2_bphi_B0=0.0;
    double Bhalo_tree_fNL2_bphi_sq_advection=0.0;
    double Bhalo_tree_fNL2_bphi_sq_F2=0.0;
    double Bhalo_tree_fNL2_bphi_sq_b2=0.0;
    double Bhalo_tree_fNL2_bphi_sq_bK2=0.0;
    double Bhalo_tree_fNL2_bphi_sq_reconstruction=0.0;
    double Bhalo_tree_fNL2_bphi_bphidelta=0.0;
    double Bhalo_tree_fNL2_bphi2_operator=0.0;
    double Bhalo_tree_fNL2_deterministic=0.0;
    double Bhalo_tree_fNL2_stochastic_alpha3PNG_basis=0.0;
    double matter_tree=0.0;
    double matter_B122I=0.0;
    double matter_B122II=0.0;
    double matter_B113I=0.0;
    double matter_B113II=0.0;
    double matter_loop=0.0;
    double matter_total=0.0;
    double matter_B122I_error_estimate=0.0;
    double matter_B122II_error_estimate=0.0;
    double matter_B113I_error_estimate=0.0;
    double matter_B113II_error_estimate=0.0;
    double matter_loop_error_estimate=0.0;
    double matter_total_error_estimate=0.0;
    std::uint64_t matter_linear_neval=0;
    double matter_B112II=0.0;
    double matter_B112II_error_bound=0.0;
    std::uint64_t matter_B112II_neval=0;
};

void add_tree(
    NativeAccumulator& sum,
    const native::ComponentResult& value,
    double weight) {
#define ADD_NATIVE_FIELD(name) \
    sum.name+=weight*value.name
    ADD_NATIVE_FIELD(stochastic_alpha3_basis);
    ADD_NATIVE_FIELD(dBdfNL_local_tree);
    ADD_NATIVE_FIELD(dBdfNL_local_primordial);
    ADD_NATIVE_FIELD(dBdfNL_local_bphi_f2);
    ADD_NATIVE_FIELD(dBdfNL_local_bphi_advection);
    ADD_NATIVE_FIELD(dBdfNL_local_bphidelta);
    ADD_NATIVE_FIELD(dBdfNL_local_bphi_b2);
    ADD_NATIVE_FIELD(dBdfNL_local_bphi_bK2);
    ADD_NATIVE_FIELD(dBdfNL_local_bphi_reconstruction);
    ADD_NATIVE_FIELD(dBdfNL_stochastic_alpha3_basis);
    ADD_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_B0);
    ADD_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_sq_advection);
    ADD_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_sq_F2);
    ADD_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_sq_b2);
    ADD_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_sq_bK2);
    ADD_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_sq_reconstruction);
    ADD_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_bphidelta);
    ADD_NATIVE_FIELD(Bhalo_tree_fNL2_bphi2_operator);
    ADD_NATIVE_FIELD(Bhalo_tree_fNL2_deterministic);
    ADD_NATIVE_FIELD(
        Bhalo_tree_fNL2_stochastic_alpha3PNG_basis);
#undef ADD_NATIVE_FIELD
}

void add_matter_linear(
    NativeAccumulator& sum,
    const native::ComponentResult& value,
    double weight) {
    sum.matter_tree+=weight*value.dBdfNL_local_tree;
    sum.matter_B122I+=weight*value.dBdfNL_local_B122I;
    sum.matter_B122II+=weight*value.dBdfNL_local_B122II;
    sum.matter_B113I+=weight*value.dBdfNL_local_B113I;
    sum.matter_B113II+=weight*value.dBdfNL_local_B113II;
    sum.matter_loop+=weight*value.dBdfNL_local_1loop;
    sum.matter_total+=weight*value.dBdfNL_local_total;
    const double absolute_weight=std::fabs(weight);
    sum.matter_B122I_error_estimate+=
        absolute_weight
        *value.dBdfNL_local_B122I_stats.abserr;
    sum.matter_B122II_error_estimate+=
        absolute_weight
        *value.dBdfNL_local_B122II_stats.abserr;
    sum.matter_B113I_error_estimate+=
        absolute_weight
        *value.dBdfNL_local_B113I_stats.abserr;
    sum.matter_B113II_error_estimate+=
        absolute_weight
        *value.dBdfNL_local_B113II_stats.abserr;
    sum.matter_loop_error_estimate+=
        absolute_weight
        *value.dBdfNL_local_1loop_stats.abserr;
    sum.matter_total_error_estimate+=
        absolute_weight
        *value.dBdfNL_local_total_stats.abserr;
    sum.matter_linear_neval+=
        static_cast<std::uint64_t>(
            std::max(
                0,
                value.dBdfNL_local_total_stats.neval));
}

void write_named_value(
    const char* name,
    double value,
    bool& first) {
    if (!first) std::cout<<',';
    first=false;
    write_json_string(name);
    std::cout<<':'<<value;
}

void write_tree(const NativeAccumulator& value) {
    std::cout<<'{';
    bool first=true;
#define WRITE_NATIVE_FIELD(name) \
    write_named_value(#name,value.name,first)
    WRITE_NATIVE_FIELD(stochastic_alpha3_basis);
    WRITE_NATIVE_FIELD(dBdfNL_local_tree);
    WRITE_NATIVE_FIELD(dBdfNL_local_primordial);
    WRITE_NATIVE_FIELD(dBdfNL_local_bphi_f2);
    WRITE_NATIVE_FIELD(dBdfNL_local_bphi_advection);
    WRITE_NATIVE_FIELD(dBdfNL_local_bphidelta);
    WRITE_NATIVE_FIELD(dBdfNL_local_bphi_b2);
    WRITE_NATIVE_FIELD(dBdfNL_local_bphi_bK2);
    WRITE_NATIVE_FIELD(dBdfNL_local_bphi_reconstruction);
    WRITE_NATIVE_FIELD(dBdfNL_stochastic_alpha3_basis);
    WRITE_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_B0);
    WRITE_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_sq_advection);
    WRITE_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_sq_F2);
    WRITE_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_sq_b2);
    WRITE_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_sq_bK2);
    WRITE_NATIVE_FIELD(
        Bhalo_tree_fNL2_bphi_sq_reconstruction);
    WRITE_NATIVE_FIELD(Bhalo_tree_fNL2_bphi_bphidelta);
    WRITE_NATIVE_FIELD(Bhalo_tree_fNL2_bphi2_operator);
    WRITE_NATIVE_FIELD(Bhalo_tree_fNL2_deterministic);
    WRITE_NATIVE_FIELD(
        Bhalo_tree_fNL2_stochastic_alpha3PNG_basis);
#undef WRITE_NATIVE_FIELD
    std::cout<<'}';
}

void write_matter_linear(const NativeAccumulator& value) {
    std::cout<<"{\"tree\":"<<value.matter_tree
             <<",\"B122I\":"<<value.matter_B122I
             <<",\"B122II\":"<<value.matter_B122II
             <<",\"B113I\":"<<value.matter_B113I
             <<",\"B113II\":"<<value.matter_B113II
             <<",\"loop\":"<<value.matter_loop
             <<",\"total\":"<<value.matter_total
             <<",\"numerical_error_estimate\":{"
             <<"\"B122I\":"
             <<value.matter_B122I_error_estimate
             <<",\"B122II\":"
             <<value.matter_B122II_error_estimate
             <<",\"B113I\":"
             <<value.matter_B113I_error_estimate
             <<",\"B113II\":"
             <<value.matter_B113II_error_estimate
             <<",\"loop\":"
             <<value.matter_loop_error_estimate
             <<",\"total\":"
             <<value.matter_total_error_estimate
             <<"},\"neval\":"
             <<value.matter_linear_neval
             <<'}';
}

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=25) {
            std::cerr
                <<"usage: eft_v2_post_r1_png_template_driver "
                <<"POWER_TABLE PNG_TABLE EDGE_FILE "
                <<"RADIAL_STOP START STOP "
                <<"SHELL_NRAD SHELL_NMU N_ALPHA N_COS_BETA N_GAMMA "
                <<"[tree-fixed|matter-linear|matter-b112] LAMBDA "
                <<"EPSREL P13_EPSREL QMIN QMAX PNG_IR_CUTOFF "
                <<"QMC_POWER QMC_REPLICATES R_SMOOTH B_REC_H "
                <<"CELL_SIZE CIC_POWER\n";
            return 2;
        }
        const std::string power_path=argv[1];
        const std::string table_path=argv[2];
        const std::string edge_path=argv[3];
        const int radial_stop=
            parse_int(argv[4],"radial stop");
        const int start=parse_int(argv[5],"start",0);
        const int stop=parse_int(argv[6],"stop");
        if (radial_stop>15
            ||start<0 ||stop<=start ||stop>120) {
            throw std::invalid_argument(
                "require radial_stop<=15 and "
                "0<=start<stop<=120");
        }
        eft::HaarOrientedLatticeShellRuleConfig shell_config;
        shell_config.exact.radial_order=
            parse_int(argv[7],"shell radial order");
        shell_config.exact.angular_order=
            parse_int(argv[8],"shell angular order");
        shell_config.exact.fft_box_size=1000.0;
        shell_config.exact.fft_mesh_size=256;
        shell_config.n_alpha=parse_int(argv[9],"alpha order");
        shell_config.n_cos_beta=
            parse_int(argv[10],"cos-beta order");
        shell_config.n_gamma=parse_int(argv[11],"gamma order");
        const std::string sector=argv[12];
        if (sector!="tree-fixed"
            &&sector!="matter-linear"
            &&sector!="matter-b112") {
            throw std::invalid_argument("invalid PNG sector");
        }
        const double lambda=parse_double(argv[13],"lambda");
        if (lambda<0.0) {
            throw std::invalid_argument(
                "lambda must be nonnegative");
        }
        native::NativeConfig native_config;
        native_config.epsrel=parse_double(argv[14],"epsrel");
        native_config.p13_epsrel=
            parse_double(argv[15],"p13 epsrel");
        native_config.qmin=parse_double(argv[16],"qmin");
        native_config.qmax=parse_double(argv[17],"qmax");
        native_config.png_ir_cutoff=
            parse_double(argv[18],"PNG IR cutoff");
        native_config.png_b112ii_multicenter_qmc=true;
        native_config.png_linear_multicenter_qmc=true;
        native_config.png_b112ii_qmc_power=
            parse_int(argv[19],"QMC power");
        native_config.png_b112ii_qmc_replicates=
            parse_int(argv[20],"QMC replicates",2);
        native_config.smoothing_radius=
            parse_double(argv[21],"smoothing radius");
        const double b_rec_h=
            parse_double(argv[22],"halo reconstruction bias");
        native_config.recon_cellsize=
            parse_double(argv[23],"cell size");
        native_config.recon_cic_window_power=
            parse_int(argv[24],"CIC power",0);
        if (!(native_config.epsrel>0.0
              &&native_config.epsrel<1.0)
            ||!(native_config.p13_epsrel>0.0
                 &&native_config.p13_epsrel<1.0)
            ||!(native_config.qmin>0.0)
            ||!(native_config.qmax>native_config.qmin)
            ||!(native_config.png_ir_cutoff>0.0)
            ||!(native_config.smoothing_radius>0.0)
            ||!(b_rec_h>0.0)
            ||native_config.recon_cellsize<0.0) {
            throw std::invalid_argument(
                "post-R1 finite-PNG integration/reconstruction "
                "parameters violate the positive-domain contract");
        }
        native_config.bias_recon=
            lambda==0.0
                ?std::numeric_limits<double>::infinity()
                :1.0/lambda;

        hv1::ReconstructionConfig fixed_reconstruction;
        fixed_reconstruction.enabled=true;
        fixed_reconstruction.smoothing_radius=
            native_config.smoothing_radius;
        fixed_reconstruction.bias_recon=b_rec_h;
        fixed_reconstruction.cell_size=
            native_config.recon_cellsize;
        eft::StochasticIntegrationConfig fixed_integration;
        fixed_integration.qmin=native_config.qmin;
        fixed_integration.qmax=native_config.qmax;
        fixed_integration.n_radial=1;
        fixed_integration.n_mu=1;
        fixed_integration.n_phi=1;
        fixed_integration.restore_p13_uv_tail=false;

        const std::vector<double> edges=read_edges(edge_path);
        const TabulatedPower power(power_path);
        const TabulatedColumn transfer(table_path,3);
        const std::string power_sha256=file_sha256(power_path);
        const std::string table_sha256=file_sha256(table_path);
        const std::string edge_sha256=file_sha256(edge_path);
        const std::string executable_sha256=file_sha256(argv[0]);
        const eft::EftBiasKernelProvider eft_base;
        const LocalPngK1Provider png_zero(
            eft_base,transfer,0.0);
        const LocalPngK1Provider png_plus(
            eft_base,transfer,1.0);
        const LocalPngK1Provider png_minus(
            eft_base,transfer,-1.0);

        std::cout<<std::setprecision(17)
                 <<"{\"record\":\"header\","
                 <<"\"schema\":"
                   "\"marisa-b-post-r1-finite-png-jsonl-v5\","
                 <<"\"sector\":";
        write_json_string(sector);
        std::cout<<",\"lambda\":"<<lambda
                 <<",\"input_contract\":{"
                 <<"\"linear_power\":"
                   "\"POWER_TABLE_column_1\","
                 <<"\"transfer_M\":"
                   "\"PNG_TABLE_column_3\"}"
                 <<",\"source_hashes\":{"
                 <<"\"linear_power\":";
        write_json_string(power_sha256);
        std::cout<<",\"png_table\":";
        write_json_string(table_sha256);
        std::cout<<",\"edge_file\":";
        write_json_string(edge_sha256);
        std::cout<<",\"driver_executable\":";
        write_json_string(executable_sha256);
        std::cout<<",\"parameter_registry\":";
        write_json_string(eft::registry_sha256());
        std::cout<<'}'
                 <<",\"lambda_definition\":"
                   "\"b1_over_b_rec_h_for_matter_uplift\","
                 <<"\"shell_projection\":{"
                 <<"\"measure\":\"exact float32 FFT-lattice "
                   "k1-k2-mu plus normalized Haar orientation "
                   "cubature\","
                 <<"\"fft_box_size\":1000,"
                 <<"\"fft_mesh_size\":256,"
                 <<"\"radial_order\":"
                 <<shell_config.exact.radial_order
                 <<",\"angular_order\":"
                 <<shell_config.exact.angular_order
                 <<",\"orientation_orders\":["
                 <<shell_config.n_alpha<<','
                 <<shell_config.n_cos_beta<<','
                 <<shell_config.n_gamma<<"]},"
                 <<"\"reconstruction\":{\"R\":"
                 <<native_config.smoothing_radius
                 <<",\"b_rec_h\":"<<b_rec_h
                 <<",\"cell_size\":"
                 <<native_config.recon_cellsize
                 <<",\"cic_power\":"
                 <<native_config.recon_cic_window_power
                 <<"},\"integration\":{\"epsrel\":"
                 <<native_config.epsrel
                 <<",\"p13_epsrel\":"
                 <<native_config.p13_epsrel
                 <<",\"qmin\":"<<native_config.qmin
                 <<",\"qmax\":"<<native_config.qmax
                 <<",\"png_ir_cutoff\":"
                 <<native_config.png_ir_cutoff
                 <<",\"png_ir_cutoff_scope\":"
                   "\"all_local_primordial_B0_T0_legs\""
                 <<",\"matter_linear_multicenter_qmc\":true"
                 <<",\"b112ii_qmc_power\":"
                 <<native_config.png_b112ii_qmc_power
                 <<",\"b112ii_qmc_replicates\":"
                 <<native_config.png_b112ii_qmc_replicates
                 <<"},\"fixed_poisson_png\":{"
                 <<"\"intensity\":\"conditional_1_plus_delta_h\","
                 <<"\"external_estimator_filter\":"
                   "\"at_most_one_density_mark_per_cumulant\","
                 <<"\"finite_K1\":"
                   "\"b1+fNL*bphi/M(k)\","
                 <<"\"bare_PL3\":false},"
                 <<"\"residual_stochastic_png\":{"
                 <<"\"shape_convention\":"
                   "\"shared_G_L_Q_for_independent_alpha3_and_alpha3PNG\","
                 <<"\"linear_shape_extraction\":"
                   "\"H_plus_minus_H_minus_over_4\","
                 <<"\"primary_forward\":"
                   "\"B_Gres_times_G_plus_fL_plus_B_PNGres_times_fL_plus_f2Q\","
                 <<"\"supports_eq2p65_two_amplitude\":true,"
                 <<"\"leading_noisy_shift\":false},"
                 <<"\"radial_stop\":"<<radial_stop
                 <<",\"bin_range\":["
                 <<start<<','<<stop<<"]}\n";
        std::cout.flush();

        int flat=0;
        for (int first=0;first<15;++first) {
            for (int second=first;
                 second<15;
                 ++second,++flat) {
                if (flat<start ||flat>=stop) continue;
                if (first>=radial_stop
                    ||second>=radial_stop) continue;
                const shell::ShellBin bin{
                    edges[static_cast<std::size_t>(first)],
                    edges[static_cast<std::size_t>(first+1)],
                    edges[static_cast<std::size_t>(second)],
                    edges[static_cast<std::size_t>(second+1)]};
                const eft::HaarOrientedLatticeShellRule rule=
                    eft::make_haar_oriented_lattice_shell_rule(
                        bin,shell_config);
                NativeAccumulator accumulated;
                eft::SparsePolynomial fixed_gaussian;
                eft::SparsePolynomial fixed_linear;
                eft::SparsePolynomial fixed_quadratic;
                eft::SparsePolynomial residual_density_gaussian;
                eft::SparsePolynomial residual_density_linear;
                eft::SparsePolynomial residual_density_quadratic;
                eft::SparsePolynomial residual_tied_gaussian;
                eft::SparsePolynomial residual_tied_linear;
                eft::SparsePolynomial residual_tied_quadratic;
                std::uint64_t fixed_generated=0;
                std::uint64_t fixed_allowed=0;
                for (const shell::ShellNode& node:rule.nodes) {
                    const auto vectors=
                        native_vectors(node.closed_vectors);
                    if (sector=="tree-fixed") {
                        native::NativeConfig halo_config=
                            native_config;
                        halo_config.bias_recon=b_rec_h;
                        native::HaloBiasV1Params unit;
                        unit.b1=1.0;
                        unit.b2=1.0;
                        unit.bK2=1.0;
                        unit.bphi=1.0;
                        unit.bphidelta=1.0;
                        unit.bphi2=1.0;
                        add_tree(
                            accumulated,
                            native::
                            compute_post_recon_halo_bias_v1_local_png_tree_dfNL_vectors(
                                power,transfer,vectors,
                                halo_config,unit),
                            node.weight);

                        const auto zero=
                            eft::reconstructed_fixed_poisson_bispectrum(
                                power,node.closed_vectors,
                                png_zero,fixed_reconstruction,
                                fixed_integration,
                                eft::ReconstructedStochasticMap::
                                    TiedDensityAndShift,
                                eft::PoissonIntensityMode::
                                    ConditionalTracer,
                                false);
                        const auto plus=
                            eft::reconstructed_fixed_poisson_bispectrum(
                                power,node.closed_vectors,
                                png_plus,fixed_reconstruction,
                                fixed_integration,
                                eft::ReconstructedStochasticMap::
                                    TiedDensityAndShift,
                                eft::PoissonIntensityMode::
                                    ConditionalTracer,
                                false);
                        const auto minus=
                            eft::reconstructed_fixed_poisson_bispectrum(
                                power,node.closed_vectors,
                                png_minus,fixed_reconstruction,
                                fixed_integration,
                                eft::ReconstructedStochasticMap::
                                    TiedDensityAndShift,
                                eft::PoissonIntensityMode::
                                    ConditionalTracer,
                                false);
                        const auto& p0=
                            zero.tree
                                .by_inverse_number_density[1];
                        const auto& pp=
                            plus.tree
                                .by_inverse_number_density[1];
                        const auto& pm=
                            minus.tree
                                .by_inverse_number_density[1];
                        fixed_gaussian+=node.weight*p0;
                        fixed_linear+=
                            node.weight*0.5*(pp-pm);
                        fixed_quadratic+=
                            node.weight*(0.5*(pp+pm)-p0);
                        const auto density_zero=
                            eft::
                            reconstructed_leading_bshot_residual_generating_shape(
                                power,node.closed_vectors,
                                png_zero,fixed_reconstruction,
                                false);
                        const auto density_plus=
                            eft::
                            reconstructed_leading_bshot_residual_generating_shape(
                                power,node.closed_vectors,
                                png_plus,fixed_reconstruction,
                                false);
                        const auto density_minus=
                            eft::
                            reconstructed_leading_bshot_residual_generating_shape(
                                power,node.closed_vectors,
                                png_minus,fixed_reconstruction,
                                false);
                        residual_density_gaussian+=
                            node.weight*density_zero;
                        residual_density_linear+=
                            node.weight*0.25
                            *(density_plus-density_minus);
                        residual_density_quadratic+=
                            node.weight
                            *(0.5
                              *(density_plus+density_minus)
                              -density_zero);
                        const auto tied_zero=
                            eft::
                            reconstructed_leading_bshot_residual_generating_shape(
                                power,node.closed_vectors,
                                png_zero,fixed_reconstruction,
                                true);
                        const auto tied_plus=
                            eft::
                            reconstructed_leading_bshot_residual_generating_shape(
                                power,node.closed_vectors,
                                png_plus,fixed_reconstruction,
                                true);
                        const auto tied_minus=
                            eft::
                            reconstructed_leading_bshot_residual_generating_shape(
                                power,node.closed_vectors,
                                png_minus,fixed_reconstruction,
                                true);
                        residual_tied_gaussian+=
                            node.weight*tied_zero;
                        residual_tied_linear+=
                            node.weight*0.25
                            *(tied_plus-tied_minus);
                        residual_tied_quadratic+=
                            node.weight
                            *(0.5*(tied_plus+tied_minus)
                              -tied_zero);
                        fixed_generated+=
                            zero.tree.generated_topologies;
                        fixed_allowed+=
                            zero.tree.estimator_allowed_topologies;
                    } else if (sector=="matter-linear") {
                        add_matter_linear(
                            accumulated,
                            native::
                            compute_post_recon_local_png_1loop_dfNL_vectors(
                                power,transfer,vectors,
                                native_config,1.0),
                            node.weight);
                    } else {
                        const auto value=
                            native::
                            compute_post_recon_local_png_B112II_fNL2_coefficient_vectors(
                                power,transfer,vectors,
                                native_config);
                        accumulated.matter_B112II+=
                            node.weight
                            *value.B112II_fNL2_coefficient;
                        accumulated.matter_B112II_error_bound+=
                            std::fabs(node.weight)
                            *value
                                .B112II_fNL2_coefficient_stats
                                .abserr;
                        accumulated.matter_B112II_neval+=
                            static_cast<std::uint64_t>(
                                std::max(
                                    0,
                                    value
                                    .B112II_fNL2_coefficient_stats
                                    .neval));
                    }
                }
                PolynomialComparison residual_native_comparison;
                PolynomialComparison residual_map_comparison;
                if (sector=="tree-fixed") {
                    const eft::SparsePolynomial b1=
                        eft::SparsePolynomial::variable(
                            eft::ParameterId::B1);
                    const std::array<
                        eft::SparsePolynomial,3> expected{{
                        accumulated.stochastic_alpha3_basis
                            *b1*b1,
                        accumulated
                            .dBdfNL_stochastic_alpha3_basis
                            *b1,
                        eft::SparsePolynomial::constant(
                            accumulated
                            .Bhalo_tree_fNL2_stochastic_alpha3PNG_basis)
                    }};
                    const std::array<
                        eft::SparsePolynomial,3> density{{
                        residual_density_gaussian,
                        residual_density_linear,
                        residual_density_quadratic
                    }};
                    const std::array<
                        eft::SparsePolynomial,3> tied{{
                        residual_tied_gaussian,
                        residual_tied_linear,
                        residual_tied_quadratic
                    }};
                    for (std::size_t order=0;
                         order<expected.size();
                         ++order) {
                        require_polynomial_close(
                            density[order],expected[order],
                            "finite-PNG residual/native G-L-Q");
                        require_polynomial_close(
                            tied[order],density[order],
                            "finite-PNG residual tied/density");
                        const PolynomialComparison native=
                            compare_polynomials(
                                density[order],
                                expected[order]);
                        residual_native_comparison
                            .maximum_absolute_error=std::max(
                                residual_native_comparison
                                    .maximum_absolute_error,
                                native.maximum_absolute_error);
                        residual_native_comparison
                            .maximum_scale=std::max(
                                residual_native_comparison
                                    .maximum_scale,
                                native.maximum_scale);
                        const PolynomialComparison maps=
                            compare_polynomials(
                                tied[order],density[order]);
                        residual_map_comparison
                            .maximum_absolute_error=std::max(
                                residual_map_comparison
                                    .maximum_absolute_error,
                                maps.maximum_absolute_error);
                        residual_map_comparison
                            .maximum_scale=std::max(
                                residual_map_comparison
                                    .maximum_scale,
                                maps.maximum_scale);
                    }
                }
                std::cout<<"{\"record\":\"bin\",\"index\":"
                         <<flat<<",\"edges\":["
                         <<bin.k1_lower<<','<<bin.k1_upper<<','
                         <<bin.k2_lower<<','<<bin.k2_upper<<"],"
                         <<"\"shell_nodes\":"
                         <<rule.nodes.size()
                         <<",\"invariant_shell_nodes\":"
                         <<rule.diagnostics.invariant_nodes
                         <<",\"orientation_nodes\":"
                         <<rule.diagnostics.orientation_nodes
                         <<",\"zero_external_leg_pairs\":"
                         <<rule.diagnostics.exact
                                .zero_external_leg_pairs
                         <<",\"closing_zero_pairs\":"
                         <<rule.diagnostics.exact
                                .closing_zero_pairs
                         <<",\"valid_pair_fraction\":"
                         <<rule.diagnostics.exact
                                .valid_pair_fraction;
                if (sector=="tree-fixed") {
                    std::cout<<",\"halo_tree\":";
                    write_tree(accumulated);
                    std::cout<<",\"fixed_poisson_png\":{"
                             <<"\"gaussian\":";
                    write_polynomial(fixed_gaussian);
                    std::cout<<",\"linear_unit_bphi\":";
                    write_polynomial(fixed_linear);
                    std::cout<<",\"quadratic_unit_bphi2\":";
                    write_polynomial(fixed_quadratic);
                    std::cout<<",\"generated_topologies_sum\":"
                             <<fixed_generated
                             <<",\"estimator_allowed_topologies_sum\":"
                             <<fixed_allowed<<'}';
                    std::cout
                        <<",\"residual_stochastic_png\":{"
                        <<"\"density_only\":";
                    write_finite_polynomial_coefficients(
                        residual_density_gaussian,
                        residual_density_linear,
                        residual_density_quadratic);
                    std::cout<<",\"tied\":";
                    write_finite_polynomial_coefficients(
                        residual_tied_gaussian,
                        residual_tied_linear,
                        residual_tied_quadratic);
                    std::cout<<",\"noisy_shift\":";
                    write_finite_polynomial_coefficients(
                        residual_tied_gaussian
                            -residual_density_gaussian,
                        residual_tied_linear
                            -residual_density_linear,
                        residual_tied_quadratic
                            -residual_density_quadratic);
                    std::cout<<'}';
                    std::cout
                        <<",\"residual_stochastic_png_gate\":{"
                        <<"\"shape_convention\":"
                          "\"shared_G_L_Q_for_independent_alpha3_and_alpha3PNG\","
                        <<"\"finite_generator_linear_extraction\":"
                          "\"H_plus_minus_H_minus_over_4\","
                        <<"\"native_max_abs\":"
                        <<residual_native_comparison
                            .maximum_absolute_error
                        <<",\"native_max_rel\":"
                        <<residual_native_comparison
                            .relative_error()
                        <<",\"tied_minus_density_max_abs\":"
                        <<residual_map_comparison
                            .maximum_absolute_error
                        <<",\"tied_minus_density_max_rel\":"
                        <<residual_map_comparison
                            .relative_error()
                        <<",\"noisy_shift_exactly_zero\":true,"
                        <<"\"supports_eq2p65_two_amplitude\":true,"
                        <<"\"legacy_single_amplitude_G_plus_fL_plus_f2Q_is_eq2p65\":false}";
                } else if (sector=="matter-linear") {
                    std::cout<<",\"matter_linear\":";
                    write_matter_linear(accumulated);
                } else {
                    std::cout<<",\"matter_b112\":{"
                             <<"\"coefficient\":"
                             <<accumulated.matter_B112II
                             <<",\"weighted_error_bound\":"
                             <<accumulated
                                .matter_B112II_error_bound
                             <<",\"neval\":"
                             <<accumulated
                                .matter_B112II_neval
                             <<'}';
                }
                std::cout<<"}\n";
                std::cout.flush();
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            <<"post R1 finite-PNG template driver failed: "
            <<error.what()<<'\n';
        return 1;
    }
}

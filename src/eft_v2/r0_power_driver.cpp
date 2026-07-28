#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PowerSpectrum.h"
#include "ir_resummation.h"
#include "tracer_power.h"

namespace eft=marisa_b_eft_v2;
namespace shell=marisa_b_shell_v1;

namespace {

class TabulatedPower final:public PowerSpectrum {
public:
    explicit TabulatedPower(const std::string& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open power table: "+path);
        std::string line;
        while (std::getline(input,line)) {
            if (line.empty() || line[0]=='#') continue;
            std::istringstream row(line);
            double k=0.0,p=0.0;
            if (row>>k>>p) {
                if (!(k>0.0 && p>0.0)) throw std::runtime_error("power table is not positive");
                log_k_.push_back(std::log(k));
                log_p_.push_back(std::log(p));
            }
        }
        if (log_k_.size()<2) throw std::runtime_error("power table is too short");
    }
    real Evaluate(real k) const override {
        if (!(k>0.0)) return 0.0;
        const double x=std::log(k);
        std::size_t right=1;
        if (x>=log_k_.back()) right=log_k_.size()-1;
        else if (x>log_k_.front()) {
            right=std::upper_bound(log_k_.begin(),log_k_.end(),x)-log_k_.begin();
        }
        const std::size_t left=right-1;
        const double fraction=(x-log_k_[left])/(log_k_[right]-log_k_[left]);
        return std::exp(log_p_[left]+fraction*(log_p_[right]-log_p_[left]));
    }
    const Cosmology& GetCosmology() const override {
        throw std::logic_error("TabulatedPower has no Cosmology object");
    }
private:
    std::vector<double> log_k_;
    std::vector<double> log_p_;
};

int integer(const char* text,const std::string& name,int minimum=1) {
    std::size_t consumed=0;
    const int value=std::stoi(text,&consumed);
    if (consumed!=std::string(text).size() || value<minimum) {
        throw std::invalid_argument("invalid integer for "+name);
    }
    return value;
}

double number(const char* text,const std::string& name) {
    std::size_t consumed=0;
    const double value=std::stod(text,&consumed);
    if (consumed!=std::string(text).size() || !std::isfinite(value)) {
        throw std::invalid_argument("invalid number for "+name);
    }
    return value;
}

std::vector<double> edges(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open P0 edge file: "+path);
    std::vector<double> result;
    std::string line;
    while (std::getline(input,line)) {
        if (line.empty() || line[0]=='#') continue;
        std::istringstream row(line);
        double value=0.0;
        if (!(row>>value)) throw std::runtime_error("invalid P0 edge row");
        result.push_back(value);
    }
    if (result.size()!=48) throw std::runtime_error("R0 P0 edge file must contain 48 values");
    return result;
}

std::vector<std::uint64_t> mode_counts(
    const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "cannot open P0 mode-count file: " + path);
    }
    std::vector<std::uint64_t> result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        unsigned long long value = 0;
        if (!(row >> value) || value == 0) {
            throw std::runtime_error(
                "invalid P0 mode-count row");
        }
        result.push_back(
            static_cast<std::uint64_t>(value));
    }
    if (result.size() != 47) {
        throw std::runtime_error(
            "R0 P0 mode-count file must contain 47 values");
    }
    return result;
}

void json_string(const std::string& value) {
    std::cout<<'"';
    for (const char character:value) {
        if (character=='"' || character=='\\') std::cout<<'\\';
        std::cout<<character;
    }
    std::cout<<'"';
}

void polynomial(const eft::SparsePolynomial& value) {
    std::cout<<'{';
    bool first=true;
    for (const auto& term:value.terms()) {
        if (!first) std::cout<<',';
        first=false;
        json_string(term.first.canonical_string());
        std::cout<<':'<<term.second;
    }
    std::cout<<'}';
}

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=17) {
            std::cerr<<"usage: eft_v2_r0_power_driver POWER_TABLE EDGE_FILE MODE_COUNT_FILE IR0_OR_1 "
                     <<"SHELL_NRAD LOOP_NRAD LOOP_NMU QMIN QMAX MU_REN LAMBDA_IR R_S "
                     <<"SIGMA_POWER_0_P11_1_PNW SPLIT_0_EH_RATIO_1_LOGK3 START STOP\n";
            return 2;
        }
        const bool use_ir=integer(argv[4],"IR flag",0)!=0;
        shell::ShellQuadratureConfig shell_config;
        shell_config.n_radial=integer(argv[5],"shell radial order");
        shell_config.radial_measure=shell::ShellRadialMeasure::FftLattice;
        shell_config.fft_box_size=1000.0;
        shell_config.fft_mesh_size=256;
        shell_config.cap_fft_lattice_radial_order_to_support=true;
        shell_config.fft_lattice_binning=
            shell::ShellFftLatticeBinning::
                EstimatorModeCountConstrainedIntegerRadius;
        eft::TracerPowerIntegrationConfig integration;
        integration.n_radial=integer(argv[6],"loop radial order");
        integration.n_mu=integer(argv[7],"loop mu order");
        integration.qmin=number(argv[8],"qmin");
        integration.qmax=number(argv[9],"qmax");
        integration.uv.mu_ren=number(argv[10],"renormalization scale");
        eft::IrResummationConfig ir_config;
        ir_config.lambda_ir=number(argv[11],"IR cutoff");
        ir_config.sound_horizon=number(argv[12],"sound horizon");
        const int sigma_mode=integer(argv[13],"Sigma power mode",0);
        const int split_mode=integer(argv[14],"smooth split mode",0);
        if (sigma_mode>1 || split_mode>1) throw std::invalid_argument("IR modes must be zero or one");
        ir_config.sigma_power=sigma_mode==0
            ?eft::SigmaPowerPrescription::FullP11:eft::SigmaPowerPrescription::NoWiggle;
        ir_config.split=split_mode==0
            ?eft::SmoothSplitPrescription::GaussianEisensteinHuRatio
            :eft::SmoothSplitPrescription::GaussianLogK3Power;
        const int start=integer(argv[15],"start",0);
        const int stop=integer(argv[16],"stop");
        if (!(start>=0 && stop>start && stop<=47)) throw std::invalid_argument("invalid P0 range");
        const auto radial_edges=edges(argv[2]);
        const auto expected_mode_counts=
            mode_counts(argv[3]);
        TabulatedPower linear(argv[1]);
        const eft::IrResummation ir(linear,ir_config);
        const PowerSpectrum& loop=use_ir
            ?static_cast<const PowerSpectrum&>(ir.loop_power()):linear;
        const PowerSpectrum& tree=use_ir
            ?static_cast<const PowerSpectrum&>(ir.tree_power()):linear;
        std::cout<<std::setprecision(17)
                 <<"{\"record\":\"header\",\"schema\":\"marisa-b-eft-v2-r0-p0-jsonl-v1\","
                 <<"\"implementation_version\":"
                 <<"\"eft-v2-p0-adaptive-estimator-constrained-radial-v2\","
                 <<"\"ir_enabled\":"<<(use_ir?"true":"false")
                 <<",\"ir_metadata\":"<<(use_ir?ir.metadata_json():"null")
                 <<",\"parameter_registry_sha256\":\""<<eft::registry_sha256()
                 <<"\",\"bin_range\":["<<start<<','<<stop<<"],\"shell_nrad\":"
                 <<shell_config.n_radial
                 <<",\"shell_projection\":{\"measure\":"
                 <<"\"integer FFT-lattice radial marginal with authoritative estimator mode-count boundary allocation\","
                 <<"\"normalization\":\"raw shell-mode average\","
                 <<"\"binning\":\"JAXPower P0 pk_nmodes-constrained lower-inclusive upper-exclusive baseline\","
                 <<"\"mode_count_source\":";
        json_string(argv[3]);
        std::cout<<','
                 <<"\"requested_radial_order\":"
                 <<shell_config.n_radial<<','
                 <<"\"cap_order_to_unique_support\":true,"
                 <<"\"box_size_mpc_h\":"
                 <<shell_config.fft_box_size<<','
                 <<"\"mesh_size\":"
                 <<shell_config.fft_mesh_size<<"},"
                 <<"\"loop_quadrature\":["
                 <<integration.n_radial<<','<<integration.n_mu<<"],\"q_range\":["
                 <<integration.qmin<<','<<integration.qmax<<"],\"mu_ren\":"
                 <<integration.uv.mu_ren
                 <<",\"p13_uv_tail_restoration\":{\"enabled\":"
                 <<(integration.restore_p13_uv_tail?"true":"false")
                 <<",\"sampled_kmax\":"<<integration.uv_tail_kmax
                 <<",\"quadrature_order\":"
                 <<integration.uv_tail_quadrature_order<<"}}\n";
        const eft::EftBiasKernelProvider provider;
        for (int index=start;index<stop;++index) {
            shell_config.fft_lattice_expected_mode_count=
                expected_mode_counts[
                    static_cast<std::size_t>(index)];
            const auto value=eft::tracer_power_shell_templates(
                loop,tree,radial_edges[index],radial_edges[index+1],
                provider,integration,shell_config);
            std::cout<<"{\"record\":\"bin\",\"index\":"<<index
                     <<",\"edges\":["<<radial_edges[index]<<','<<radial_edges[index+1]
                     <<"],\"k_effective\":"<<value.k
                     <<",\"shell_nodes\":"<<value.shell_nodes
                     <<",\"shell_mode_count\":"
                     <<value.shell_mode_count
                     <<",\"shell_expected_mode_count\":"
                     <<expected_mode_counts[
                            static_cast<std::size_t>(index)]
                     <<",\"shell_boundary_mode_adjustment\":"
                     <<value.shell_boundary_mode_adjustment
                     <<",\"shell_unique_radius_count\":"
                     <<value.shell_unique_radius_count
                     <<",\"tree\":";
            polynomial(value.tree);
            std::cout<<",\"P22\":"; polynomial(value.P22);
            std::cout<<",\"P13\":"; polynomial(value.P13);
            std::cout<<",\"P22_bare\":"; polynomial(value.P22_bare);
            std::cout<<",\"P13_bare\":"; polynomial(value.P13_bare);
            std::cout<<",\"P22_stochastic_subtraction\":";
            polynomial(value.P22_stochastic_subtraction);
            std::cout<<",\"P13_bias_subtraction\":"; polynomial(value.P13_bias_subtraction);
            std::cout<<",\"P13_uv_tail_restoration\":";
            polynomial(value.P13_uv_tail_restoration);
            std::cout<<",\"counterterm_bnabla2_delta\":";
            polynomial(value.counterterm_bnabla2_delta);
            std::cout<<",\"stochastic_pshot\":"<<value.stochastic_pshot
                     <<",\"stochastic_a0\":"<<value.stochastic_a0
                     <<",\"integration_nodes\":"<<value.integration_nodes<<"}\n";
            std::cout.flush();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"EFT-v2 R0 power driver failed: "<<error.what()<<"\n";
        return 1;
    }
}

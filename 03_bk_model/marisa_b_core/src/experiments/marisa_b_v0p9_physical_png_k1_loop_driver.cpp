#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PowerSpectrum.h"
#include "field_kernel_provider.h"
#include "parameter_registry.h"
#include "shell_projector.h"

namespace eft=marisa_b_eft_v2;
namespace hv1=marisa_b_halo_v1;
namespace shell=marisa_b_shell_v1;

namespace {

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

class TabulatedPower final:public PowerSpectrum {
public:
    explicit TabulatedPower(const std::string& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open power table: "+path);
        std::string line;
        while (std::getline(input,line)) {
            if (line.empty() ||line[0]=='#') continue;
            std::istringstream row(line);
            double k=0.0;
            double power=0.0;
            if (!(row>>k>>power) ||!(k>0.0 &&power>0.0)) {
                throw std::runtime_error("invalid positive power-table row");
            }
            log_k_.push_back(std::log(k));
            log_value_.push_back(std::log(power));
        }
        if (log_k_.size()<2) throw std::runtime_error("power table is too short");
    }

    real Evaluate(real k) const override {
        if (!(k>0.0)) return 0.0;
        const double x=std::log(k);
        std::size_t right=1;
        if (x>=log_k_.back()) {
            right=log_k_.size()-1;
        } else if (x>log_k_.front()) {
            right=static_cast<std::size_t>(
                std::upper_bound(log_k_.begin(),log_k_.end(),x)-log_k_.begin());
        }
        const std::size_t left=right-1;
        const double fraction=(x-log_k_[left])/(log_k_[right]-log_k_[left]);
        return std::exp(
            log_value_[left]+fraction*(log_value_[right]-log_value_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("TabulatedPower has no Cosmology object");
    }

private:
    std::vector<double> log_k_;
    std::vector<double> log_value_;
};

class TabulatedTransfer final:public PowerSpectrum {
public:
    explicit TabulatedTransfer(const std::string& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("cannot open local-PNG transfer table: "+path);
        }
        std::string line;
        while (std::getline(input,line)) {
            if (line.empty() ||line[0]=='#') continue;
            std::istringstream row(line);
            double k=0.0;
            double power=0.0;
            double ignored=0.0;
            double transfer=0.0;
            if (!(row>>k>>power>>ignored>>transfer)
                ||!(k>0.0 &&transfer>0.0)) {
                throw std::runtime_error("invalid local-PNG transfer-table row");
            }
            log_k_.push_back(std::log(k));
            log_value_.push_back(std::log(transfer));
        }
        if (log_k_.size()<2) {
            throw std::runtime_error("local-PNG transfer table is too short");
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
                std::upper_bound(log_k_.begin(),log_k_.end(),x)-log_k_.begin());
        }
        const std::size_t left=right-1;
        const double fraction=(x-log_k_[left])/(log_k_[right]-log_k_[left]);
        return std::exp(
            log_value_[left]+fraction*(log_value_[right]-log_value_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error("TabulatedTransfer has no Cosmology object");
    }

private:
    std::vector<double> log_k_;
    std::vector<double> log_value_;
};

/*
 * This provider exactly matches the released v0.9 Gaussian uplift:
 * every unreconstructed SPT matter kernel carries one common b1 factor.
 * Only K1 receives the physical local-PNG perturbation.  Gamma31 is an
 * auxiliary automatic-differentiation marker, not a fitted bias operator:
 *
 *   K1 = b1 + gamma31 / M(k),        k >= k_fundamental.
 *
 * Passing this provider through ReconstructedFieldKernelProvider propagates
 * the same perturbation through both the displaced density and its shift,
 * while reconstruction.bias_recon remains a fixed catalogue constant.
 */
class PhysicalPngTaggedMatterProvider final:public eft::FieldKernelProvider {
public:
    PhysicalPngTaggedMatterProvider(
        const PowerSpectrum& transfer,double minimum_k)
        :transfer_(transfer),minimum_k_(minimum_k) {}

    eft::SparsePolynomial deterministic(
        const std::vector<eft::Vec3>& momenta) const override {
        eft::SparsePolynomial result=
            eft::spt_F(momenta)
            *eft::SparsePolynomial::variable(eft::ParameterId::B1);
        if (momenta.size()==1) {
            const double k=hv1::norm(momenta.front());
            if (k>=minimum_k_) {
                const double transfer=transfer_(k);
                if (!(transfer>0.0)) {
                    throw std::runtime_error("non-positive local-PNG transfer");
                }
                result+=(1.0/transfer)
                    *eft::SparsePolynomial::variable(eft::ParameterId::Gamma31);
            }
        }
        return result;
    }

    std::string_view name() const noexcept override {
        return "v0p9_linear_bias_matter_with_physical_png_K1_marker";
    }

private:
    const PowerSpectrum& transfer_;
    double minimum_k_;
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

void write_named_polynomial(
    const char* name,const eft::SparsePolynomial& value,bool& first) {
    if (!first) std::cout<<',';
    first=false;
    write_json_string(name);
    std::cout<<':';
    write_polynomial(value);
}

void write_bin(int index,const eft::EftShellTemplates& value) {
    std::cout<<std::setprecision(17)
             <<"{\"record\":\"bin\",\"index\":"<<index
             <<",\"edges\":["<<value.bin.k1_lower<<','<<value.bin.k1_upper
             <<','<<value.bin.k2_lower<<','<<value.bin.k2_upper<<']'
             <<",\"shell_nodes\":"<<value.shell_nodes
             <<",\"invariant_shell_nodes\":"<<value.invariant_shell_nodes
             <<",\"orientation_nodes\":"<<value.orientation_nodes
             <<",\"total_loop_nodes\":"<<value.total_loop_nodes
             <<",\"zero_external_leg_pairs\":"<<value.zero_external_leg_pairs
             <<",\"closing_zero_pairs\":"<<value.closing_zero_pairs
             <<",\"diagrams\":{";
    bool first=true;
    write_named_polynomial("tree",value.diagrams.tree,first);
    write_named_polynomial("B222",value.diagrams.B222,first);
    write_named_polynomial("B321I",value.diagrams.B321I,first);
    write_named_polynomial("B321II",value.diagrams.B321II,first);
    write_named_polynomial("B411",value.diagrams.B411,first);
    write_named_polynomial("B321II_bare",value.diagrams.B321II_bare,first);
    write_named_polynomial("B411_bare",value.diagrams.B411_bare,first);
    write_named_polynomial(
        "B321II_uv_subtraction",value.diagrams.B321II_uv_subtraction,first);
    write_named_polynomial(
        "B411_uv_subtraction",value.diagrams.B411_uv_subtraction,first);
    write_named_polynomial(
        "B321II_uv_restoration",value.diagrams.B321II_uv_restoration,first);
    write_named_polynomial(
        "B411_uv_restoration",value.diagrams.B411_uv_restoration,first);
    write_named_polynomial("one_loop",value.diagrams.one_loop,first);
    write_named_polynomial("total",value.diagrams.total,first);
    std::cout<<"}}\n";
    std::cout.flush();
}

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=23) {
            std::cerr
                <<"usage: marisa_b_v0p9_physical_png_k1_loop_driver "
                <<"POWER_TABLE PNG_TABLE EDGE_FILE RADIAL_STOP START STOP "
                <<"SHELL_NRAD SHELL_NMU N_ALPHA N_COS_BETA N_GAMMA "
                <<"LOOP_NRAD LOOP_NMU LOOP_NPHI QMIN QMAX UV_TAIL_KMAX "
                <<"R_SMOOTH B_REC CELL_SIZE KMIN "
                <<"[tensor-gauss|fibonacci]\n";
            return 2;
        }
        const std::string power_path=argv[1];
        const std::string transfer_path=argv[2];
        const std::string edge_path=argv[3];
        const int radial_stop=parse_int(argv[4],"radial stop");
        const int start=parse_int(argv[5],"start",0);
        const int stop=parse_int(argv[6],"stop");
        if (radial_stop>15 ||start<0 ||stop<=start ||stop>120) {
            throw std::invalid_argument("require radial_stop<=15 and 0<=start<stop<=120");
        }

        eft::ReconstructedShellIntegrationConfig integration;
        integration.shell.exact.radial_order=parse_int(argv[7],"shell radial order");
        integration.shell.exact.angular_order=parse_int(argv[8],"shell angular order");
        integration.shell.exact.fft_box_size=1000.0;
        integration.shell.exact.fft_mesh_size=256;
        integration.shell.n_alpha=parse_int(argv[9],"alpha order");
        integration.shell.n_cos_beta=parse_int(argv[10],"cos-beta order");
        integration.shell.n_gamma=parse_int(argv[11],"gamma order");
        integration.loop.n_radial=parse_int(argv[12],"loop radial order");
        integration.loop.n_mu=parse_int(argv[13],"loop mu order");
        integration.loop.n_phi=parse_int(argv[14],"loop phi order");
        integration.loop.qmin=parse_double(argv[15],"qmin");
        integration.loop.qmax=parse_double(argv[16],"qmax");
        integration.loop.restore_uv_tail=true;
        integration.loop.uv_tail_kmax=parse_double(argv[17],"UV tail kmax");
        integration.loop.ir_safe=true;
        integration.loop.uv_subtract=true;
        integration.loop.exploit_phi_reflection=false;
        const std::string angular_rule=argv[22];
        if (angular_rule=="tensor-gauss") {
            integration.loop.angular_rule=eft::DirectAngularRule::TensorGaussLegendre;
        } else if (angular_rule=="fibonacci") {
            integration.loop.angular_rule=eft::DirectAngularRule::AntipodalFibonacci;
        } else {
            throw std::invalid_argument("angular rule must be tensor-gauss or fibonacci");
        }
        integration.include_reconstructed_stochastic=false;
        integration.include_reconstructed_fixed_poisson=false;
        integration.include_bare_fixed_poisson_one_loop=false;

        hv1::ReconstructionConfig reconstruction;
        reconstruction.enabled=true;
        reconstruction.smoothing_radius=parse_double(argv[18],"smoothing radius");
        reconstruction.bias_recon=parse_double(argv[19],"reconstruction bias");
        reconstruction.cell_size=parse_double(argv[20],"cell size");
        const double minimum_k=parse_double(argv[21],"minimum PNG wavenumber");
        if (!(integration.loop.qmin>0.0)
            ||!(integration.loop.qmax>integration.loop.qmin)
            ||!(integration.loop.uv_tail_kmax>integration.loop.qmax)
            ||!(reconstruction.smoothing_radius>0.0)
            ||!(reconstruction.bias_recon>0.0)
            ||reconstruction.cell_size<0.0
            ||!(minimum_k>0.0)) {
            throw std::invalid_argument("integration/reconstruction domain contract failed");
        }

        eft::StochasticIntegrationConfig stochastic;
        stochastic.qmin=integration.loop.qmin;
        stochastic.qmax=integration.loop.qmax;
        stochastic.n_radial=integration.loop.n_radial;
        stochastic.n_mu=integration.loop.n_mu;
        stochastic.n_phi=integration.loop.n_phi;
        stochastic.restore_p13_uv_tail=true;
        stochastic.uv_tail_kmax=integration.loop.uv_tail_kmax;

        const TabulatedPower power(power_path);
        const TabulatedTransfer transfer(transfer_path);
        const std::vector<double> edges=read_edges(edge_path);
        const PhysicalPngTaggedMatterProvider provider(transfer,minimum_k);

        std::cout<<std::setprecision(17)
                 <<"{\"record\":\"header\","
                 <<"\"schema\":\"marisa-b-v0p9-physical-png-k1-loop-marker-v1\","
                 <<"\"model\":\"v0.9 pure-linear-bias matter SPT loops with physical local-PNG K1 marker\","
                 <<"\"production_candidate\":false,"
                 <<"\"response_definition\":\"coefficient linear in gamma31 at gamma31=0; multiply by physical bphi\","
                 <<"\"gamma31_is_auxiliary_marker\":true,"
                 <<"\"gaussian_architecture\":\"each unreconstructed matter Fn carries one b1; no b2 or tidal loop operators\","
                 <<"\"fixed_b_rec\":true,"
                 <<"\"bin_range\":["<<start<<','<<stop<<"],"
                 <<"\"radial_stop\":"<<radial_stop<<','
                 <<"\"reconstruction\":{\"R\":"<<reconstruction.smoothing_radius
                 <<",\"b_rec_h\":"<<reconstruction.bias_recon
                 <<",\"cell_size\":"<<reconstruction.cell_size
                 <<",\"kmin\":"<<minimum_k<<"},"
                 <<"\"shell_orders\":["<<integration.shell.exact.radial_order<<','
                 <<integration.shell.exact.angular_order<<','
                 <<integration.shell.n_alpha<<','<<integration.shell.n_cos_beta<<','
                 <<integration.shell.n_gamma<<"],"
                 <<"\"loop_orders\":["<<integration.loop.n_radial<<','
                 <<integration.loop.n_mu<<','<<integration.loop.n_phi<<"],"
                 <<"\"q_range\":["<<integration.loop.qmin<<','
                 <<integration.loop.qmax<<"],"
                 <<"\"uv_tail_kmax\":"<<integration.loop.uv_tail_kmax<<','
                 <<"\"loop_angular_rule\":";
        write_json_string(eft::direct_angular_rule_name(integration.loop.angular_rule));
        std::cout<<",\"source_hashes\":{\"linear_power\":";
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
                const eft::EftShellTemplates value=
                    eft::compute_reconstructed_eft_shell_templates(
                        power,power,bin,provider,reconstruction,
                        integration,stochastic,0.30);
                write_bin(flat,value);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr<<"v0.9 physical PNG K1 loop driver failed: "
                 <<error.what()<<'\n';
        return 1;
    }
}

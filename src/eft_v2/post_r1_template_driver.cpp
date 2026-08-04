#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PowerSpectrum.h"
#include "parameter_registry.h"
#include "shell_projector.h"

namespace eft=marisa_b_eft_v2;
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
                "cannot open power table: "+path);
        }
        std::string line;
        while (std::getline(input,line)) {
            if (line.empty() || line[0]=='#') continue;
            std::istringstream row(line);
            double k=0.0;
            double p=0.0;
            if (!(row>>k>>p) || !(k>0.0 && p>0.0)) {
                throw std::runtime_error(
                    "invalid positive power-table row");
            }
            log_k_.push_back(std::log(k));
            log_p_.push_back(std::log(p));
        }
        if (log_k_.size()<2) {
            throw std::runtime_error(
                "power table is too short");
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
            log_p_[left]
            +fraction*(log_p_[right]-log_p_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error(
            "TabulatedPower has no Cosmology object");
    }

private:
    std::vector<double> log_k_;
    std::vector<double> log_p_;
};

class TabulatedTransfer final:public PowerSpectrum {
public:
    explicit TabulatedTransfer(const std::string& path) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error(
                "cannot open local-PNG transfer table: "+path);
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
                throw std::runtime_error(
                    "invalid local-PNG transfer-table row");
            }
            log_k_.push_back(std::log(k));
            log_transfer_.push_back(std::log(transfer));
        }
        if (log_k_.size()<2) {
            throw std::runtime_error(
                "local-PNG transfer table is too short");
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
            log_transfer_[left]
            +fraction*(log_transfer_[right]
                       -log_transfer_[left]));
    }

    const Cosmology& GetCosmology() const override {
        throw std::logic_error(
            "TabulatedTransfer has no Cosmology object");
    }

private:
    std::vector<double> log_k_;
    std::vector<double> log_transfer_;
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
    for (std::size_t index=1;
         index<result.size();
         ++index) {
        if (!(result[index]>result[index-1])) {
            throw std::runtime_error(
                "post R1 radial edges must be increasing");
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

void write_named_polynomial(
    const char* name,
    const eft::SparsePolynomial& value,
    bool& first) {
    if (!first) std::cout<<',';
    first=false;
    write_json_string(name);
    std::cout<<':';
    write_polynomial(value);
}

void write_stochastic_block(
    const eft::StochasticTemplates& stochastic) {
    std::cout<<'{';
    bool first=true;
    for (std::size_t parameter=0;
         parameter<eft::kStochasticCount;
         ++parameter) {
        if (!first) std::cout<<',';
        first=false;
        write_json_string(
            std::string(
                eft::parameter_info(
                    eft::stochastic_parameter_ids()[
                        parameter]).name));
        std::cout<<':';
        write_polynomial(
            stochastic.raw[parameter]);
    }
    std::cout<<'}';
}

void write_fixed_poisson_order(
    const eft::FixedPoissonOrderTemplates& value) {
    std::cout<<"{\"by_inverse_number_density\":{";
    for (std::size_t inverse_nbar=1;
         inverse_nbar<4;
         ++inverse_nbar) {
        if (inverse_nbar>1) std::cout<<',';
        write_json_string(
            "nbar^-"+std::to_string(inverse_nbar));
        std::cout<<':';
        write_polynomial(
            value.by_inverse_number_density[
                inverse_nbar]);
    }
    std::cout<<"},\"generated_topologies\":"
             <<value.generated_topologies
             <<",\"estimator_allowed_topologies\":"
             <<value.estimator_allowed_topologies
             <<",\"integration_nodes\":"
             <<value.integration_nodes
             <<'}';
}

void write_bin(
    int index,
    const eft::EftShellTemplates& value) {
    std::cout<<std::setprecision(17)
             <<"{\"record\":\"bin\",\"index\":"
             <<index<<",\"edges\":["
             <<value.bin.k1_lower<<','
             <<value.bin.k1_upper<<','
             <<value.bin.k2_lower<<','
             <<value.bin.k2_upper<<"],"
             <<"\"shell_nodes\":"<<value.shell_nodes<<','
             <<"\"invariant_shell_nodes\":"
             <<value.invariant_shell_nodes<<','
             <<"\"orientation_nodes\":"
             <<value.orientation_nodes<<','
             <<"\"total_loop_nodes\":"
             <<value.total_loop_nodes<<','
             <<"\"exact_joint_lattice_measure\":"
             <<(value.exact_joint_lattice_measure
                    ?"true":"false")<<','
             <<"\"exact_k1_k2_mu_lattice_measure\":"
             <<(value.exact_k1_k2_mu_lattice_measure
                    ?"true":"false")<<','
             <<"\"conditional_cubic_orientation_exact\":"
             <<(value.conditional_cubic_orientation_exact
                    ?"true":"false")<<','
             <<"\"haar_oriented_cic_projection\":"
             <<(value.haar_oriented_cic_projection
                    ?"true":"false")<<','
             <<"\"reconstructed_counterterms\":"
             <<(value.reconstructed_counterterms
                    ?"true":"false")<<','
             <<"\"reconstructed_stochastic\":"
             <<(value.reconstructed_stochastic
                    ?"true":"false")<<','
             <<"\"reconstructed_stochastic_decomposed\":"
             <<(value.reconstructed_stochastic_decomposed
                    ?"true":"false")<<','
             <<"\"reconstructed_fixed_poisson\":"
             <<(value.reconstructed_fixed_poisson
                    ?"true":"false")<<','
             <<"\"exact_lattice_radial_order\":"
             <<value.exact_lattice_radial_order<<','
             <<"\"exact_lattice_angular_order\":"
             <<value.exact_lattice_angular_order<<','
             <<"\"zero_external_leg_pairs\":"
             <<value.zero_external_leg_pairs<<','
             <<"\"closing_zero_pairs\":"
             <<value.closing_zero_pairs<<','
             <<"\"exact_lattice_valid_pair_fraction\":"
             <<value.exact_lattice_valid_pair_fraction<<','
             <<"\"exact_lattice_total_variation\":"
             <<value.exact_lattice_total_variation<<','
             <<"\"diagrams\":{";
    bool first=true;
    write_named_polynomial(
        "tree",value.diagrams.tree,first);
    write_named_polynomial(
        "B222",value.diagrams.B222,first);
    write_named_polynomial(
        "B321I",value.diagrams.B321I,first);
    write_named_polynomial(
        "B321II",value.diagrams.B321II,first);
    write_named_polynomial(
        "B411",value.diagrams.B411,first);
    write_named_polynomial(
        "B321II_bare",
        value.diagrams.B321II_bare,first);
    write_named_polynomial(
        "B411_bare",
        value.diagrams.B411_bare,first);
    write_named_polynomial(
        "B321II_uv_subtraction",
        value.diagrams.B321II_uv_subtraction,first);
    write_named_polynomial(
        "B411_uv_subtraction",
        value.diagrams.B411_uv_subtraction,first);
    write_named_polynomial(
        "B321II_uv_restoration",
        value.diagrams.B321II_uv_restoration,first);
    write_named_polynomial(
        "B411_uv_restoration",
        value.diagrams.B411_uv_restoration,first);
    std::cout<<"},\"counterterms\":{";
    first=true;
    for (std::size_t parameter=0;
         parameter<eft::kCountertermCount;
         ++parameter) {
        if (!first) std::cout<<',';
        first=false;
        write_json_string(
            std::string(
                eft::parameter_info(
                    eft::counterterm_parameter_ids()[
                        parameter]).name));
        std::cout<<':';
        write_polynomial(
            value.counterterms.total[parameter]);
    }
    std::cout<<"},\"stochastic\":";
    write_stochastic_block(value.stochastic);
    std::cout<<",\"stochastic_density_only\":";
    write_stochastic_block(
        value.stochastic_density_only);
    std::cout<<",\"stochastic_noisy_shift\":";
    write_stochastic_block(
        value.stochastic_noisy_shift);
    std::cout<<",\"bshot_bnabla2_cross\":";
    write_polynomial(
        value.stochastic.bshot_bnabla2_cross);
    std::cout<<",\"bshot_bnabla2_cross_density_only\":";
    write_polynomial(
        value.stochastic_density_only
            .bshot_bnabla2_cross);
    std::cout<<",\"bshot_bnabla2_cross_noisy_shift\":";
    write_polynomial(
        value.stochastic_noisy_shift
            .bshot_bnabla2_cross);
    std::cout<<",\"fixed_poisson\":{\"tree\":";
    write_fixed_poisson_order(
        value.fixed_poisson.tree);
    std::cout<<",\"one_loop\":";
    write_fixed_poisson_order(
        value.fixed_poisson.one_loop);
    std::cout<<'}';
    std::cout<<",\"stochastic_status\":"
             <<"\"estimator_matched_fixed_conditional_P_over_nbar"
               "_plus_released_renormalized_stochastic_PL3_closure"
               "_bare_fixed_PL3_excluded\""
             <<"}\n";
    std::cout.flush();
}

}  // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=21 &&argc!=25) {
            std::cerr
                <<"usage: eft_v2_post_r1_template_driver "
                <<"POWER_TABLE EDGE_FILE RADIAL_INDEX_STOP "
                <<"START_BIN STOP_BIN "
                <<"SHELL_NRAD SHELL_NMU "
                <<"N_ALPHA N_COS_BETA N_GAMMA "
                <<"LOOP_NRAD LOOP_NMU LOOP_NPHI "
                <<"QMIN QMAX UV_TAIL_KMAX "
                <<"R_SMOOTH B_REC CELL_SIZE "
                <<"[tensor-gauss|fibonacci] "
                <<"[PNG_TABLE FNL_REC BPHI_REC KREC_MIN]\n";
            return 2;
        }
        const bool adaptive_brec=argc==25;
        const std::string power_path=argv[1];
        const std::string edge_path=argv[2];
        const int radial_stop=
            parse_int(argv[3],"radial index stop");
        const int start=
            parse_int(argv[4],"start bin",0);
        const int stop=
            parse_int(argv[5],"stop bin");
        if (radial_stop>15
            ||start<0 ||stop<=start ||stop>120) {
            throw std::invalid_argument(
                "require radial_stop<=15 and "
                "0<=start<stop<=120");
        }

        eft::ReconstructedShellIntegrationConfig integration;
        integration.shell.exact.radial_order=
            parse_int(argv[6],"shell radial order");
        integration.shell.exact.angular_order=
            parse_int(argv[7],"shell angular order");
        integration.shell.exact.fft_box_size=1000.0;
        integration.shell.exact.fft_mesh_size=256;
        integration.shell.n_alpha=
            parse_int(argv[8],"alpha order");
        integration.shell.n_cos_beta=
            parse_int(argv[9],"cos-beta order");
        integration.shell.n_gamma=
            parse_int(argv[10],"gamma order");
        integration.loop.n_radial=
            parse_int(argv[11],"loop radial order");
        integration.loop.n_mu=
            parse_int(argv[12],"loop mu order");
        integration.loop.n_phi=
            parse_int(argv[13],"loop phi order");
        integration.loop.qmin=
            parse_double(argv[14],"qmin");
        integration.loop.qmax=
            parse_double(argv[15],"qmax");
        integration.loop.restore_uv_tail=true;
        integration.loop.uv_tail_kmax=
            parse_double(argv[16],"UV tail kmax");
        integration.loop.ir_safe=true;
        integration.loop.uv_subtract=true;
        integration.loop.exploit_phi_reflection=false;
        const std::string angular_rule=argv[20];
        if (angular_rule=="tensor-gauss") {
            integration.loop.angular_rule=
                eft::DirectAngularRule::TensorGaussLegendre;
        } else if (angular_rule=="fibonacci") {
            integration.loop.angular_rule=
                eft::DirectAngularRule::AntipodalFibonacci;
        } else {
            throw std::invalid_argument(
                "angular rule must be tensor-gauss or fibonacci");
        }
        integration.include_reconstructed_stochastic=true;
        integration.include_reconstructed_fixed_poisson=true;
        integration.include_bare_fixed_poisson_one_loop=false;
        eft::StochasticIntegrationConfig stochastic;
        stochastic.qmin=integration.loop.qmin;
        stochastic.qmax=integration.loop.qmax;
        stochastic.n_radial=integration.loop.n_radial;
        stochastic.n_mu=integration.loop.n_mu;
        stochastic.n_phi=integration.loop.n_phi;
        stochastic.restore_p13_uv_tail=true;
        stochastic.uv_tail_kmax=
            integration.loop.uv_tail_kmax;

        marisa_b_halo_v1::ReconstructionConfig reconstruction;
        reconstruction.enabled=true;
        reconstruction.smoothing_radius=
            parse_double(argv[17],"smoothing radius");
        reconstruction.bias_recon=
            parse_double(argv[18],"reconstruction bias");
        reconstruction.cell_size=
            parse_double(argv[19],"cell size");
        const std::string png_table_path=
            adaptive_brec?argv[21]:"";
        const double fnl_rec=adaptive_brec
            ?parse_double(argv[22],"fNL_rec"):0.0;
        const double bphi_rec=adaptive_brec
            ?parse_double(argv[23],"bphi_rec"):0.0;
        const double krec_min=adaptive_brec
            ?parse_double(argv[24],"reconstruction minimum k"):0.0;
        if (!(integration.loop.qmin>0.0)
            ||!(integration.loop.qmax>integration.loop.qmin)
            ||!(integration.loop.uv_tail_kmax
                 >integration.loop.qmax)
            ||!(reconstruction.smoothing_radius>0.0)
            ||!(reconstruction.bias_recon>0.0)
            ||reconstruction.cell_size<0.0
            ||!(krec_min>=0.0)) {
            throw std::invalid_argument(
                "post-R1 Gaussian integration/reconstruction "
                "parameters violate the positive-domain contract");
        }

        const std::vector<double> edges=
            read_edges(edge_path);
        const TabulatedPower power(power_path);
        std::unique_ptr<TabulatedTransfer> transfer;
        if (adaptive_brec) {
            transfer=std::make_unique<TabulatedTransfer>(
                png_table_path);
            reconstruction.local_png_bias_transfer=
                transfer.get();
            reconstruction.local_png_bias_amplitude=
                fnl_rec*bphi_rec;
            reconstruction.local_png_bias_kmin=krec_min;
        }
        const std::string power_sha256=file_sha256(power_path);
        const std::string edge_sha256=file_sha256(edge_path);
        const std::string executable_sha256=file_sha256(argv[0]);
        const std::string png_table_sha256=
            adaptive_brec?file_sha256(png_table_path):"";
        const eft::EftBiasKernelProvider base;
        std::cout<<std::setprecision(17)
                 <<"{\"record\":\"header\","
                 <<"\"schema\":";
        write_json_string(
            adaptive_brec
            ?"marisa-b-eft-v2-post-r1-adaptive-brec-finite-jsonl-v1"
            :"marisa-b-eft-v2-post-r1-jsonl-v4");
        std::cout<<","
                 <<"\"model\":\"post-recon halo Gaussian one-loop "
                   "EFT-v2 deterministic plus reconstructed "
                   "counterterms, estimator-matched fixed Poisson, "
                   "and residual stochastic closure\","
                 <<"\"ir_resummation\":false,"
                 <<"\"production_candidate\":"
                 <<(adaptive_brec?"false":"true")<<','
                 <<"\"renormalization\":{\"status\":"
                   "\"renormalized-direct-post-r1-candidate\","
                 <<"\"uv_subtraction\":true,"
                 <<"\"subleading_tail_restoration\":true,"
                 <<"\"tail_kmax\":"
                 <<integration.loop.uv_tail_kmax<<"},"
                 <<"\"counterterm_convention\":{"
                 <<"\"bispectrum\":\"Bakx-eq-Bk-ctr\","
                 <<"\"total\":\"BctrI+BctrII\","
                 <<"\"field_direction_derivative\":"
                   "\"BctrI+2*BctrII\","
                 <<"\"shared_b_nabla2_with_power\":"
                   "\"local-in-time evolution assumption\"},"
                 <<"\"reconstruction\":{\"R\":"
                 <<reconstruction.smoothing_radius
                 <<",\"b_rec_h\":"
                 <<reconstruction.bias_recon
                 <<",\"cell_size\":"
                 <<reconstruction.cell_size
                 <<",\"cic_power\":4";
        if (adaptive_brec) {
            std::cout
                <<",\"adaptive_local_png_bias\":{"
                <<"\"fNL_rec\":"<<fnl_rec
                <<",\"bphi_rec\":"<<bphi_rec
                <<",\"amplitude\":"
                <<reconstruction.local_png_bias_amplitude
                <<",\"kmin\":"<<krec_min
                <<",\"transfer_column\":3}";
        }
        std::cout<<"},"
                 <<"\"shell_projection\":{\"measure\":"
                 <<"\"exact float32 FFT-lattice k1-k2-mu "
                   "plus normalized Haar orientation cubature\","
                 <<"\"fft_box_size\":1000,"
                 <<"\"fft_mesh_size\":256,"
                 <<"\"conditional_cubic_orientation_exact\":false,"
                 <<"\"direct_cubic_orbit_validation_required\":true,"
                 <<"\"normalization\":\"raw N1*N2 estimator pairs\","
                 <<"\"radial_order\":"
                 <<integration.shell.exact.radial_order
                 <<",\"angular_order\":"
                 <<integration.shell.exact.angular_order
                 <<",\"orientation_orders\":["
                 <<integration.shell.n_alpha<<','
                 <<integration.shell.n_cos_beta<<','
                 <<integration.shell.n_gamma<<"]},"
                 <<"\"loop_quadrature\":["
                 <<integration.loop.n_radial<<','
                 <<integration.loop.n_mu<<','
                 <<integration.loop.n_phi<<"],"
                 <<"\"loop_angular_rule\":";
        write_json_string(
            eft::direct_angular_rule_name(
                integration.loop.angular_rule));
        std::cout<<",\"q_range\":["
                 <<integration.loop.qmin<<','
                 <<integration.loop.qmax<<"],"
                 <<"\"fixed_poisson\":{"
                 <<"\"intensity\":\"conditional_1_plus_delta_h\","
                 <<"\"external_estimator_filter\":"
                   "\"at_most_one_density_mark_per_cumulant\","
                 <<"\"leading_P_over_nbar\":true,"
                 <<"\"bare_PL3_in_production\":false,"
                 <<"\"bare_PL3_reason\":"
                   "\"requires independent UV renormalization\"},"
                 <<"\"stochastic_status\":"
                 <<"\"estimator_matched_fixed_conditional_P_over_nbar"
                   "_plus_released_renormalized_stochastic_PL3_closure"
                   "_bare_fixed_PL3_excluded\","
                 <<"\"stochastic_coordinate\":"
                 <<"\"JAXPower external-output S111/S122/S121/S113 "
                   "self-contractions filtered per Poisson cumulant; "
                   "internal shift noise retained; Pshot fixed to zero\","
                 <<"\"source_hashes\":{\"linear_power\":";
        write_json_string(power_sha256);
        std::cout<<",\"edge_file\":";
        write_json_string(edge_sha256);
        std::cout<<",\"driver_executable\":";
        write_json_string(executable_sha256);
        if (adaptive_brec) {
            std::cout<<",\"png_table\":";
            write_json_string(png_table_sha256);
        }
        std::cout<<"},\"parameter_registry_sha256\":";
        write_json_string(eft::registry_sha256());
        std::cout<<",\"radial_index_stop\":"
                 <<radial_stop
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
                    ||second>=radial_stop) {
                    continue;
                }
                const shell::ShellBin bin{
                    edges[static_cast<std::size_t>(first)],
                    edges[static_cast<std::size_t>(first+1)],
                    edges[static_cast<std::size_t>(second)],
                    edges[static_cast<std::size_t>(second+1)]};
                const eft::EftShellTemplates value=
                    eft::compute_reconstructed_eft_shell_templates(
                        power,power,bin,base,reconstruction,
                        integration,stochastic,0.30);
                write_bin(flat,value);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr
            <<"EFT-v2 post R1 template driver failed: "
            <<error.what()<<'\n';
        return 1;
    }
}

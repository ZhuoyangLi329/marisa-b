/*
 * MARISA-B 命令行入口大纲：解析参数与三角形 -> 读取 P_L/PNG 表 -> 选择互不
 * 干扰的 DM 或 halo backend -> C++ 计算各 diagram -> 输出可审计 JSON。
 * 新增 halo bias-v1 mode 只调用 native 核心的新入口，旧 v0 mode 保持原样。
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "BSPT.h"
#include "Cosmology.h"
#include "InterpolatedPS.h"
#include "SpecialFunctions.h"
#include "array.h"
#include "marisa_b_native.h"

namespace {

struct Triangle {
    double k1 = 0.05;
    double k2 = 0.05;
    double mu12 = -0.5;
};

struct Options {
    std::string pk_table;
    std::vector<Triangle> triangles;
    double epsrel = 1.0e-3;
    double p13_epsrel = 1.0e-2;
    double qmin = 1.0e-4;
    double qmax = 30.0;
    double png_ir_cutoff = 0.0;
    std::string b112ii_integrator = "adaptive";
    int b112ii_qmc_power = 12;
    int b112ii_qmc_replicates = 4;
    double fnl = 0.0;
    double smoothing_radius = 10.0;
    double bias_recon = 1.0;
    double recon_cellsize = 0.0;
    int recon_cic_window_power = 0;
    double singular_floor = 1.0e-5;
    double ir_sigma2 = -1.0;
    double b1 = 1.0;
    double b2 = 0.0;
    double bK2 = 0.0;
    double bphi = 0.0;
    double bphidelta = 0.0;
    double bphi2 = 0.0;
    std::string mode = "pre_recon_gaussian";
    std::string backend = "react_gr_bootstrap";
    std::string png_table;
    std::string nowiggle_table;
};

struct ComponentResult {
    Triangle triangle;
    double k3 = 0.0;
    double btree = 0.0;
    double b222 = 0.0;
    double b321i = 0.0;
    double b321ii = 0.0;
    double b411 = 0.0;
    double bloopterms = 0.0;
    double b1loop = 0.0;
    double btotal = 0.0;
    double dbdfnl_local_tree = 0.0;
    double dbdfnl_local_B122I = 0.0;
    double dbdfnl_local_B122II = 0.0;
    double dbdfnl_local_B113I = 0.0;
    double dbdfnl_local_B113II = 0.0;
    double b112ii_fnl2_coefficient = 0.0;
    double dbdfnl_local_B222 = 0.0;
    double dbdfnl_local_B321I = 0.0;
    double dbdfnl_local_B321II = 0.0;
    double dbdfnl_local_B411 = 0.0;
    double dbdfnl_local_1loop = 0.0;
    double dbdfnl_local_total = 0.0;
    double dbdfnl_local_primordial = 0.0;
    double dbdfnl_local_bphi_f2 = 0.0;
    double dbdfnl_local_bphi_advection = 0.0;
    double dbdfnl_local_bphidelta = 0.0;
    double dbdfnl_local_bphi_b2 = 0.0;
    double dbdfnl_local_bphi_bK2 = 0.0;
    double dbdfnl_local_bphi_reconstruction = 0.0;
    double bhalo_tree_fnl2_bphi_B0 = 0.0;
    double bhalo_tree_fnl2_bphi_sq_advection = 0.0;
    double bhalo_tree_fnl2_bphi_sq_F2 = 0.0;
    double bhalo_tree_fnl2_bphi_sq_b2 = 0.0;
    double bhalo_tree_fnl2_bphi_sq_bK2 = 0.0;
    double bhalo_tree_fnl2_bphi_sq_reconstruction = 0.0;
    double bhalo_tree_fnl2_bphi_bphidelta = 0.0;
    double bhalo_tree_fnl2_bphi2_operator = 0.0;
    double bhalo_tree_fnl2_deterministic = 0.0;
    double bhalo_tree_fnl2_stochastic_alpha3PNG_basis = 0.0;
    double stochastic_alpha3_basis = 0.0;
    double stochastic_alpha4_basis = 1.0;
    double dbdfnl_stochastic_alpha3_basis = 0.0;
    double b222_abserr = 0.0;
    double b321i_abserr = 0.0;
    double b411_abserr = 0.0;
    double p13_k1_abserr = 0.0;
    double p13_k2_abserr = 0.0;
    double p13_k3_abserr = 0.0;
    double p12_png_k1_abserr = 0.0;
    double p12_png_k2_abserr = 0.0;
    double p12_png_k3_abserr = 0.0;
    double dbdfnl_local_B122II_abserr = 0.0;
    double dbdfnl_local_B113II_abserr = 0.0;
    double b112ii_fnl2_coefficient_abserr = 0.0;
    double dbdfnl_local_B222_abserr = 0.0;
    double dbdfnl_local_B321I_abserr = 0.0;
    double dbdfnl_local_B321II_abserr = 0.0;
    double dbdfnl_local_B411_abserr = 0.0;
    double dp13dfnl_local_k1_abserr = 0.0;
    double dp13dfnl_local_k2_abserr = 0.0;
    double dp13dfnl_local_k3_abserr = 0.0;
    int b222_neval = 0;
    int b321i_neval = 0;
    int b411_neval = 0;
    int p13_k1_neval = 0;
    int p13_k2_neval = 0;
    int p13_k3_neval = 0;
    int p12_png_k1_neval = 0;
    int p12_png_k2_neval = 0;
    int p12_png_k3_neval = 0;
    int dbdfnl_local_B122II_neval = 0;
    int dbdfnl_local_B113II_neval = 0;
    int b112ii_fnl2_coefficient_neval = 0;
    int dbdfnl_local_B222_neval = 0;
    int dbdfnl_local_B321I_neval = 0;
    int dbdfnl_local_B321II_neval = 0;
    int dbdfnl_local_B411_neval = 0;
    int dp13dfnl_local_k1_neval = 0;
    int dp13dfnl_local_k2_neval = 0;
    int dp13dfnl_local_k3_neval = 0;
};

std::vector<std::string> split(const std::string& value, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, sep)) out.push_back(item);
    return out;
}

Triangle parse_triangle(const std::string& value) {
    const std::vector<std::string> parts = split(value, ',');
    if (parts.size() != 3) {
        throw std::runtime_error("--triangle must be formatted as k1,k2,mu12");
    }
    Triangle tri;
    tri.k1 = std::atof(parts[0].c_str());
    tri.k2 = std::atof(parts[1].c_str());
    tri.mu12 = std::atof(parts[2].c_str());
    return tri;
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--pk-table" && i + 1 < argc) {
            opts.pk_table = argv[++i];
        } else if (arg == "--triangle" && i + 1 < argc) {
            opts.triangles.push_back(parse_triangle(argv[++i]));
        } else if (arg == "--epsrel" && i + 1 < argc) {
            opts.epsrel = std::atof(argv[++i]);
        } else if (arg == "--p13-epsrel" && i + 1 < argc) {
            opts.p13_epsrel = std::atof(argv[++i]);
        } else if (arg == "--qmin" && i + 1 < argc) {
            opts.qmin = std::atof(argv[++i]);
        } else if (arg == "--qmax" && i + 1 < argc) {
            opts.qmax = std::atof(argv[++i]);
        } else if (arg == "--png-ir-cutoff" && i + 1 < argc) {
            opts.png_ir_cutoff = std::atof(argv[++i]);
        } else if (arg == "--b112ii-integrator" && i + 1 < argc) {
            opts.b112ii_integrator = argv[++i];
        } else if (arg == "--b112ii-qmc-power" && i + 1 < argc) {
            opts.b112ii_qmc_power = std::atoi(argv[++i]);
        } else if (arg == "--b112ii-qmc-replicates" && i + 1 < argc) {
            opts.b112ii_qmc_replicates = std::atoi(argv[++i]);
        } else if (arg == "--fnl" && i + 1 < argc) {
            opts.fnl = std::atof(argv[++i]);
        } else if (arg == "--smoothing-radius" && i + 1 < argc) {
            opts.smoothing_radius = std::atof(argv[++i]);
        } else if (arg == "--bias-recon" && i + 1 < argc) {
            opts.bias_recon = std::atof(argv[++i]);
        } else if (arg == "--recon-cellsize" && i + 1 < argc) {
            opts.recon_cellsize = std::atof(argv[++i]);
        } else if (arg == "--recon-cic-window-power" && i + 1 < argc) {
            opts.recon_cic_window_power = std::atoi(argv[++i]);
        } else if (arg == "--singular-floor" && i + 1 < argc) {
            opts.singular_floor = std::atof(argv[++i]);
        } else if (arg == "--ir-sigma2" && i + 1 < argc) {
            opts.ir_sigma2 = std::atof(argv[++i]);
        } else if (arg == "--b1" && i + 1 < argc) {
            opts.b1 = std::atof(argv[++i]);
        } else if (arg == "--b2" && i + 1 < argc) {
            opts.b2 = std::atof(argv[++i]);
        } else if (arg == "--bk2" && i + 1 < argc) {
            opts.bK2 = std::atof(argv[++i]);
        } else if (arg == "--bphi" && i + 1 < argc) {
            opts.bphi = std::atof(argv[++i]);
        } else if (arg == "--bphidelta" && i + 1 < argc) {
            opts.bphidelta = std::atof(argv[++i]);
        } else if (arg == "--bphi2" && i + 1 < argc) {
            opts.bphi2 = std::atof(argv[++i]);
        } else if (arg == "--mode" && i + 1 < argc) {
            opts.mode = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            opts.backend = argv[++i];
        } else if (arg == "--png-table" && i + 1 < argc) {
            opts.png_table = argv[++i];
        } else if (arg == "--nowiggle-table" && i + 1 < argc) {
            opts.nowiggle_table = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: marisa_b_triangle --pk-table PATH [--triangle k1,k2,mu12] "
                << "[--epsrel 1e-3] [--p13-epsrel 1e-2] [--qmin 1e-4] [--qmax 30] "
                << "[--png-ir-cutoff 0] "
                << "[--b112ii-integrator adaptive|multicenter_qmc] "
                << "[--b112ii-qmc-power 12] [--b112ii-qmc-replicates 4] "
                << "[--fnl 0] "
                << "[--recon-cellsize 0] [--recon-cic-window-power 0] "
                << "[--mode pre_recon_gaussian|post_recon_gaussian|post_recon_gaussian_ir_nowiggle|pre_recon_local_png_tree|pre_recon_local_png_1loop|post_recon_local_png_1loop|pre_recon_local_png_b112ii_diagnostic|post_recon_local_png_b112ii_diagnostic|pre_recon_local_png_finite_1loop|post_recon_local_png_finite_1loop|pre_recon_halo_bias_v1_gaussian|post_recon_halo_bias_v1_gaussian|pre_recon_halo_bias_v1_local_png_tree|post_recon_halo_bias_v1_local_png_tree|pre_recon_halo_bias_v1_local_png_1loop_truncated|post_recon_halo_bias_v1_local_png_1loop_truncated] "
                << "[--backend react_gr_bootstrap|native_cpp] [--png-table PATH] [--nowiggle-table PATH] [--ir-sigma2 VALUE] [--b1 1] [--b2 0] [--bk2 0] [--bphi 0] [--bphidelta 0] [--bphi2 0]\n";
            std::exit(0);
        } else {
            std::stringstream msg;
            msg << "Unknown or incomplete argument: " << arg;
            throw std::runtime_error(msg.str());
        }
    }
    if (opts.pk_table.empty()) throw std::runtime_error("--pk-table is required");
    if (opts.triangles.empty()) opts.triangles.push_back(parse_triangle("0.05,0.05,-0.5"));
    if (opts.mode != "pre_recon_gaussian" && opts.mode != "post_recon_gaussian" && opts.mode != "post_recon_gaussian_ir_nowiggle" && opts.mode != "pre_recon_local_png_tree" && opts.mode != "pre_recon_local_png_1loop" && opts.mode != "post_recon_local_png_1loop" && opts.mode != "pre_recon_local_png_b112ii_diagnostic" && opts.mode != "post_recon_local_png_b112ii_diagnostic" && opts.mode != "pre_recon_local_png_finite_1loop" && opts.mode != "post_recon_local_png_finite_1loop" && opts.mode != "pre_recon_halo_bias_v1_gaussian" && opts.mode != "post_recon_halo_bias_v1_gaussian" && opts.mode != "pre_recon_halo_bias_v1_local_png_tree" && opts.mode != "post_recon_halo_bias_v1_local_png_tree" && opts.mode != "pre_recon_halo_bias_v1_local_png_1loop_truncated" && opts.mode != "post_recon_halo_bias_v1_local_png_1loop_truncated") {
        throw std::runtime_error("MARISA-B supports the frozen v0 modes plus the halo bias-v1 Gaussian, local-PNG tree, and explicitly truncated one-loop modes");
    }
    if (opts.backend != "react_gr_bootstrap" && opts.backend != "native_cpp") {
        throw std::runtime_error("MARISA-B v0 supports --backend react_gr_bootstrap or native_cpp");
    }
    if ((opts.mode == "pre_recon_local_png_tree" || opts.mode == "pre_recon_local_png_1loop" || opts.mode == "post_recon_local_png_1loop" || opts.mode == "pre_recon_local_png_b112ii_diagnostic" || opts.mode == "post_recon_local_png_b112ii_diagnostic" || opts.mode == "pre_recon_local_png_finite_1loop" || opts.mode == "post_recon_local_png_finite_1loop") && opts.backend != "native_cpp") {
        throw std::runtime_error("dark-matter local-PNG modes require --backend native_cpp");
    }
    if ((opts.mode == "post_recon_gaussian" || opts.mode == "post_recon_gaussian_ir_nowiggle") && opts.backend != "native_cpp") {
        throw std::runtime_error("post_recon_gaussian and post_recon_gaussian_ir_nowiggle require --backend native_cpp");
    }
    if ((opts.mode == "pre_recon_halo_bias_v1_gaussian" || opts.mode == "post_recon_halo_bias_v1_gaussian") && opts.backend != "native_cpp") {
        throw std::runtime_error("halo bias-v1 Gaussian modes require --backend native_cpp");
    }
    if ((opts.mode == "pre_recon_halo_bias_v1_local_png_tree" || opts.mode == "post_recon_halo_bias_v1_local_png_tree" || opts.mode == "pre_recon_halo_bias_v1_local_png_1loop_truncated" || opts.mode == "post_recon_halo_bias_v1_local_png_1loop_truncated") && opts.backend != "native_cpp") {
        throw std::runtime_error("halo bias-v1 local-PNG modes require --backend native_cpp");
    }
    if (opts.mode == "post_recon_gaussian_ir_nowiggle" && opts.nowiggle_table.empty()) {
        throw std::runtime_error("post_recon_gaussian_ir_nowiggle requires --nowiggle-table");
    }
    if (opts.mode == "post_recon_gaussian_ir_nowiggle" && opts.ir_sigma2 < 0.0) {
        throw std::runtime_error("post_recon_gaussian_ir_nowiggle requires non-negative --ir-sigma2");
    }
    if ((opts.mode == "pre_recon_local_png_tree" || opts.mode == "pre_recon_local_png_1loop" || opts.mode == "post_recon_local_png_1loop" || opts.mode == "pre_recon_local_png_b112ii_diagnostic" || opts.mode == "post_recon_local_png_b112ii_diagnostic" || opts.mode == "pre_recon_local_png_finite_1loop" || opts.mode == "post_recon_local_png_finite_1loop" || opts.mode == "pre_recon_halo_bias_v1_local_png_tree" || opts.mode == "post_recon_halo_bias_v1_local_png_tree" || opts.mode == "pre_recon_halo_bias_v1_local_png_1loop_truncated" || opts.mode == "post_recon_halo_bias_v1_local_png_1loop_truncated") && opts.png_table.empty()) {
        opts.png_table = opts.pk_table;
    }
    if (opts.png_ir_cutoff < 0.0) {
        throw std::runtime_error("--png-ir-cutoff must be non-negative");
    }
    if (!std::isfinite(opts.fnl)) {
        throw std::runtime_error("--fnl must be finite");
    }
    if ((opts.mode == "pre_recon_local_png_finite_1loop"
         || opts.mode == "post_recon_local_png_finite_1loop")
        && std::fabs(opts.b1 - 1.0) > 1.0e-14) {
        throw std::runtime_error(
            "finite-fNL dark-matter modes require --b1 1");
    }
    if (opts.b112ii_integrator != "adaptive"
        && opts.b112ii_integrator != "multicenter_qmc") {
        throw std::runtime_error(
            "--b112ii-integrator must be adaptive or multicenter_qmc");
    }
    if (opts.b112ii_qmc_power < 4 || opts.b112ii_qmc_power > 20) {
        throw std::runtime_error("--b112ii-qmc-power must lie in [4,20]");
    }
    if (opts.b112ii_qmc_replicates < 2
        || opts.b112ii_qmc_replicates > 32) {
        throw std::runtime_error(
            "--b112ii-qmc-replicates must lie in [2,32]");
    }
    return opts;
}

void read_pk_table(const std::string& path, array& k_arr, array& p_arr) {
    std::ifstream input(path.c_str());
    if (!input) {
        std::stringstream msg;
        msg << "Could not open P(k) table: " << path;
        throw std::runtime_error(msg.str());
    }

    std::vector<double> kvals;
    std::vector<double> pvals;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        double k = 0.0;
        double p = 0.0;
        if (ss >> k >> p) {
            kvals.push_back(k);
            pvals.push_back(p);
        }
    }
    if (kvals.size() < 8 || kvals.size() != pvals.size()) {
        throw std::runtime_error("P(k) table must contain at least 8 rows with two numeric columns.");
    }

    k_arr = array(kvals);
    p_arr = array(pvals);
}

void read_png_transfer_table(const std::string& path, array& k_arr, array& p_arr, array& p_phi_arr, array& m_arr) {
    std::ifstream input(path.c_str());
    if (!input) {
        std::stringstream msg;
        msg << "Could not open PNG transfer table: " << path;
        throw std::runtime_error(msg.str());
    }

    std::vector<double> kvals;
    std::vector<double> pvals;
    std::vector<double> pphivals;
    std::vector<double> mvals;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        double k = 0.0;
        double p = 0.0;
        double p_phi = 0.0;
        double m = 0.0;
        if (ss >> k >> p >> p_phi >> m) {
            kvals.push_back(k);
            pvals.push_back(p);
            pphivals.push_back(p_phi);
            mvals.push_back(m);
        }
    }
    if (kvals.size() < 8 || kvals.size() != pvals.size() || kvals.size() != pphivals.size() || kvals.size() != mvals.size()) {
        throw std::runtime_error("PNG transfer table must contain at least 8 rows with columns k, P_L, P_phi, M.");
    }

    k_arr = array(kvals);
    p_arr = array(pvals);
    p_phi_arr = array(pphivals);
    m_arr = array(mvals);
}

void set_unity_growth() {
    Dl_spt = 1.0;
    D_spt = 1.0;
    dnorm_spt = 1.0;
    fl_spt = 1.0;
    fdgp_spt = 1.0;
}

double triangle_k3(const Triangle& tri) {
    return std::sqrt(tri.k1 * tri.k1 + tri.k2 * tri.k2 + 2.0 * tri.k1 * tri.k2 * tri.mu12);
}

class BispectrumBackend {
  public:
    virtual ~BispectrumBackend() {}
    virtual ComponentResult compute(const Triangle& tri) const = 0;
};

class ReactGrBootstrapBackend : public BispectrumBackend {
  public:
    explicit ReactGrBootstrapBackend(const BSPT& bspt) : bspt_(bspt) {}

    ComponentResult compute(const Triangle& tri) const override {
        ComponentResult out;
        out.triangle = tri;
        out.k3 = triangle_k3(tri);
        out.btree = bspt_.Btree(1, tri.k1, tri.k2, tri.mu12);
        out.btotal = bspt_.Bloop(1, tri.k1, tri.k2, tri.mu12);
        out.bloopterms = bspt_.Bloopterms(1, tri.k1, tri.k2, out.k3, tri.mu12);
        bspt_.BlooptermComponentsGR(tri.k1, tri.k2, out.k3, tri.mu12, out.b222, out.b321i, out.b411);
        out.b321ii = out.btotal - out.btree - out.bloopterms;
        out.b1loop = out.btotal - out.btree;
        return out;
    }

  private:
    const BSPT& bspt_;
};

class NativeCppBackend : public BispectrumBackend {
  public:
    NativeCppBackend(const PowerSpectrum& p_l, const Options& opts) : p_l_(p_l) {
        config_.qmin = opts.qmin;
        config_.qmax = opts.qmax;
        config_.epsrel = opts.epsrel;
        config_.p13_epsrel = opts.p13_epsrel;
        config_.smoothing_radius = opts.smoothing_radius;
        config_.bias_recon = opts.bias_recon;
        config_.recon_cellsize = opts.recon_cellsize;
        config_.recon_cic_window_power = opts.recon_cic_window_power;
        config_.singular_floor = opts.singular_floor;
    }

    ComponentResult compute(const Triangle& tri) const override {
        marisa_b_native::Triangle native_tri;
        native_tri.k1 = tri.k1;
        native_tri.k2 = tri.k2;
        native_tri.mu12 = tri.mu12;
        const marisa_b_native::ComponentResult native = marisa_b_native::compute_pre_recon_gaussian(p_l_, native_tri, config_);

        ComponentResult out;
        out.triangle = tri;
        out.k3 = native.k3;
        out.btree = native.Btree;
        out.b222 = native.B222;
        out.b321i = native.B321I;
        out.b321ii = native.B321II;
        out.b411 = native.B411;
        out.bloopterms = native.Bloopterms;
        out.b1loop = native.B1loop;
        out.btotal = native.Btotal;
        out.b222_abserr = native.B222_stats.abserr;
        out.b321i_abserr = native.B321I_stats.abserr;
        out.b411_abserr = native.B411_stats.abserr;
        out.p13_k1_abserr = native.P13_k1_stats.abserr;
        out.p13_k2_abserr = native.P13_k2_stats.abserr;
        out.p13_k3_abserr = native.P13_k3_stats.abserr;
        out.b222_neval = native.B222_stats.neval;
        out.b321i_neval = native.B321I_stats.neval;
        out.b411_neval = native.B411_stats.neval;
        out.p13_k1_neval = native.P13_k1_stats.neval;
        out.p13_k2_neval = native.P13_k2_stats.neval;
        out.p13_k3_neval = native.P13_k3_stats.neval;
        return out;
    }

  private:
    const PowerSpectrum& p_l_;
    marisa_b_native::NativeConfig config_;
};

class NativePostReconBackend : public BispectrumBackend {
  public:
    NativePostReconBackend(const PowerSpectrum& p_l, const Options& opts) : p_l_(p_l) {
        config_.qmin = opts.qmin;
        config_.qmax = opts.qmax;
        config_.epsrel = opts.epsrel;
        config_.p13_epsrel = opts.p13_epsrel;
        config_.smoothing_radius = opts.smoothing_radius;
        config_.bias_recon = opts.bias_recon;
        config_.recon_cellsize = opts.recon_cellsize;
        config_.recon_cic_window_power = opts.recon_cic_window_power;
        config_.singular_floor = opts.singular_floor;
    }

    ComponentResult compute(const Triangle& tri) const override {
        marisa_b_native::Triangle native_tri;
        native_tri.k1 = tri.k1;
        native_tri.k2 = tri.k2;
        native_tri.mu12 = tri.mu12;
        const marisa_b_native::ComponentResult native = marisa_b_native::compute_post_recon_gaussian(p_l_, native_tri, config_);

        ComponentResult out;
        out.triangle = tri;
        out.k3 = native.k3;
        out.btree = native.Btree;
        out.b222 = native.B222;
        out.b321i = native.B321I;
        out.b321ii = native.B321II;
        out.b411 = native.B411;
        out.bloopterms = native.Bloopterms;
        out.b1loop = native.B1loop;
        out.btotal = native.Btotal;
        out.b222_abserr = native.B222_stats.abserr;
        out.b321i_abserr = native.B321I_stats.abserr;
        out.b411_abserr = native.B411_stats.abserr;
        out.p13_k1_abserr = native.P13_k1_stats.abserr;
        out.p13_k2_abserr = native.P13_k2_stats.abserr;
        out.p13_k3_abserr = native.P13_k3_stats.abserr;
        out.b222_neval = native.B222_stats.neval;
        out.b321i_neval = native.B321I_stats.neval;
        out.b411_neval = native.B411_stats.neval;
        out.p13_k1_neval = native.P13_k1_stats.neval;
        out.p13_k2_neval = native.P13_k2_stats.neval;
        out.p13_k3_neval = native.P13_k3_stats.neval;
        return out;
    }

  private:
    const PowerSpectrum& p_l_;
    marisa_b_native::NativeConfig config_;
};

/*
 * halo bias-v1 Gaussian backend。
 * 构造参数 p_l/opts 分别提供线性谱和 b1,b2,bK2/reconstruction 设置；
 * post_recon=false/true 选择 pre/post 新入口。compute 返回统一 JSON schema
 * 使用的 diagram 分量，且不会调用或改变旧 DM backend。
 */
class NativeHaloBiasV1GaussianBackend : public BispectrumBackend {
  public:
    NativeHaloBiasV1GaussianBackend(const PowerSpectrum& p_l, const Options& opts, bool post_recon)
        : p_l_(p_l), post_recon_(post_recon) {
        config_.qmin = opts.qmin;
        config_.qmax = opts.qmax;
        config_.epsrel = opts.epsrel;
        config_.p13_epsrel = opts.p13_epsrel;
        config_.smoothing_radius = opts.smoothing_radius;
        config_.bias_recon = opts.bias_recon;
        config_.recon_cellsize = opts.recon_cellsize;
        config_.recon_cic_window_power = opts.recon_cic_window_power;
        config_.singular_floor = opts.singular_floor;
        bias_.b1 = opts.b1;
        bias_.b2 = opts.b2;
        bias_.bK2 = opts.bK2;
    }

    ComponentResult compute(const Triangle& tri) const override {
        marisa_b_native::Triangle native_tri;
        native_tri.k1 = tri.k1;
        native_tri.k2 = tri.k2;
        native_tri.mu12 = tri.mu12;
        const marisa_b_native::ComponentResult native = post_recon_
            ? marisa_b_native::compute_post_recon_halo_bias_v1_gaussian(p_l_, native_tri, config_, bias_)
            : marisa_b_native::compute_pre_recon_halo_bias_v1_gaussian(p_l_, native_tri, config_, bias_);

        ComponentResult out;
        out.triangle = tri;
        out.k3 = native.k3;
        out.btree = native.Btree;
        out.b222 = native.B222;
        out.b321i = native.B321I;
        out.b321ii = native.B321II;
        out.b411 = native.B411;
        out.bloopterms = native.Bloopterms;
        out.b1loop = native.B1loop;
        out.btotal = native.Btotal;
        out.b222_abserr = native.B222_stats.abserr;
        out.b321i_abserr = native.B321I_stats.abserr;
        out.b411_abserr = native.B411_stats.abserr;
        out.p13_k1_abserr = native.P13_k1_stats.abserr;
        out.p13_k2_abserr = native.P13_k2_stats.abserr;
        out.p13_k3_abserr = native.P13_k3_stats.abserr;
        out.b222_neval = native.B222_stats.neval;
        out.b321i_neval = native.B321I_stats.neval;
        out.b411_neval = native.B411_stats.neval;
        out.p13_k1_neval = native.P13_k1_stats.neval;
        out.p13_k2_neval = native.P13_k2_stats.neval;
        out.p13_k3_neval = native.P13_k3_stats.neval;
        out.stochastic_alpha3_basis = native.stochastic_alpha3_basis;
        out.stochastic_alpha4_basis = native.stochastic_alpha4_basis;
        return out;
    }

  private:
    const PowerSpectrum& p_l_;
    bool post_recon_ = false;
    marisa_b_native::NativeConfig config_;
    marisa_b_native::HaloBiasV1Params bias_;
};

/*
 * Barreira-2022 pre/post-recon local-PNG tree backend。
 * 输入 P_L、M(k) 与五个 bias 系数；compute 不做 loop，只返回 Gaussian tree、
 * O(fNL) 的逐物理来源响应及 alpha3/alpha4 的未归一化 stochastic basis。
 */
class NativeHaloBiasV1PngTreeBackend : public BispectrumBackend {
  public:
    NativeHaloBiasV1PngTreeBackend(
        const PowerSpectrum& p_l,
        const PowerSpectrum& transfer_m,
        const Options& opts,
        bool post_recon)
        : p_l_(p_l), transfer_m_(transfer_m), post_recon_(post_recon) {
        config_.smoothing_radius = opts.smoothing_radius;
        config_.bias_recon = opts.bias_recon;
        config_.recon_cellsize = opts.recon_cellsize;
        config_.recon_cic_window_power = opts.recon_cic_window_power;
        bias_.b1 = opts.b1;
        bias_.b2 = opts.b2;
        bias_.bK2 = opts.bK2;
        bias_.bphi = opts.bphi;
        bias_.bphidelta = opts.bphidelta;
        bias_.bphi2 = opts.bphi2;
    }

    ComponentResult compute(const Triangle& tri) const override {
        marisa_b_native::Triangle native_tri;
        native_tri.k1 = tri.k1;
        native_tri.k2 = tri.k2;
        native_tri.mu12 = tri.mu12;
        const marisa_b_native::ComponentResult native = post_recon_
            ? marisa_b_native::compute_post_recon_halo_bias_v1_local_png_tree_dfNL(
                  p_l_, transfer_m_, native_tri, config_, bias_)
            : marisa_b_native::compute_pre_recon_halo_bias_v1_local_png_tree_dfNL(
                  p_l_, transfer_m_, native_tri, bias_);

        ComponentResult out;
        out.triangle = tri;
        out.k3 = native.k3;
        out.btree = native.Btree;
        out.btotal = native.Btotal;
        out.dbdfnl_local_tree = native.dBdfNL_local_tree;
        out.dbdfnl_local_total = native.dBdfNL_local_total;
        out.dbdfnl_local_primordial = native.dBdfNL_local_primordial;
        out.dbdfnl_local_bphi_f2 = native.dBdfNL_local_bphi_f2;
        out.dbdfnl_local_bphi_advection = native.dBdfNL_local_bphi_advection;
        out.dbdfnl_local_bphidelta = native.dBdfNL_local_bphidelta;
        out.dbdfnl_local_bphi_b2 = native.dBdfNL_local_bphi_b2;
        out.dbdfnl_local_bphi_bK2 = native.dBdfNL_local_bphi_bK2;
        out.dbdfnl_local_bphi_reconstruction = native.dBdfNL_local_bphi_reconstruction;
        out.bhalo_tree_fnl2_bphi_B0 = native.Bhalo_tree_fNL2_bphi_B0;
        out.bhalo_tree_fnl2_bphi_sq_advection =
            native.Bhalo_tree_fNL2_bphi_sq_advection;
        out.bhalo_tree_fnl2_bphi_sq_F2 =
            native.Bhalo_tree_fNL2_bphi_sq_F2;
        out.bhalo_tree_fnl2_bphi_sq_b2 =
            native.Bhalo_tree_fNL2_bphi_sq_b2;
        out.bhalo_tree_fnl2_bphi_sq_bK2 =
            native.Bhalo_tree_fNL2_bphi_sq_bK2;
        out.bhalo_tree_fnl2_bphi_sq_reconstruction =
            native.Bhalo_tree_fNL2_bphi_sq_reconstruction;
        out.bhalo_tree_fnl2_bphi_bphidelta =
            native.Bhalo_tree_fNL2_bphi_bphidelta;
        out.bhalo_tree_fnl2_bphi2_operator =
            native.Bhalo_tree_fNL2_bphi2_operator;
        out.bhalo_tree_fnl2_deterministic =
            native.Bhalo_tree_fNL2_deterministic;
        out.bhalo_tree_fnl2_stochastic_alpha3PNG_basis =
            native.Bhalo_tree_fNL2_stochastic_alpha3PNG_basis;
        out.stochastic_alpha3_basis = native.stochastic_alpha3_basis;
        out.stochastic_alpha4_basis = native.stochastic_alpha4_basis;
        out.dbdfnl_stochastic_alpha3_basis = native.dBdfNL_stochastic_alpha3_basis;
        return out;
    }

  private:
    const PowerSpectrum& p_l_;
    const PowerSpectrum& transfer_m_;
    bool post_recon_ = false;
    marisa_b_native::NativeConfig config_;
    marisa_b_native::HaloBiasV1Params bias_;
};

/*
 * halo bias-v1 local-PNG 的固定 cutoff、未重整化一环 continuation。
 *
 * 该 backend 只调用新的 KIC+A+C kernel-response 路径，不再叠加旧 matter
 * PNG 的显式 B0 loop，避免 primordial response 被重复计数。post_recon
 * 选择 product-rule reconstruction 映射；所有积分设置原样传给 native core。
 */
class NativeHaloBiasV1PngOneLoopTruncatedBackend : public BispectrumBackend {
  public:
    NativeHaloBiasV1PngOneLoopTruncatedBackend(
        const PowerSpectrum& p_l,
        const PowerSpectrum& transfer_m,
        const Options& opts,
        bool post_recon)
        : p_l_(p_l), transfer_m_(transfer_m), post_recon_(post_recon) {
        config_.qmin = opts.qmin;
        config_.qmax = opts.qmax;
        config_.epsrel = opts.epsrel;
        config_.p13_epsrel = opts.p13_epsrel;
        config_.smoothing_radius = opts.smoothing_radius;
        config_.bias_recon = opts.bias_recon;
        config_.recon_cellsize = opts.recon_cellsize;
        config_.recon_cic_window_power = opts.recon_cic_window_power;
        config_.singular_floor = opts.singular_floor;
        bias_.b1 = opts.b1;
        bias_.b2 = opts.b2;
        bias_.bK2 = opts.bK2;
        bias_.bphi = opts.bphi;
        bias_.bphidelta = opts.bphidelta;
        bias_.bphi2 = opts.bphi2;
    }

    ComponentResult compute(const Triangle& tri) const override {
        marisa_b_native::Triangle native_tri;
        native_tri.k1 = tri.k1;
        native_tri.k2 = tri.k2;
        native_tri.mu12 = tri.mu12;
        const marisa_b_native::ComponentResult native = post_recon_
            ? marisa_b_native::compute_post_recon_halo_bias_v1_local_png_1loop_truncated_dfNL(
                  p_l_, transfer_m_, native_tri, config_, bias_)
            : marisa_b_native::compute_pre_recon_halo_bias_v1_local_png_1loop_truncated_dfNL(
                  p_l_, transfer_m_, native_tri, config_, bias_);

        ComponentResult out;
        out.triangle = tri;
        out.k3 = native.k3;
        out.btree = native.Btree;
        out.b222 = native.B222;
        out.b321i = native.B321I;
        out.b321ii = native.B321II;
        out.b411 = native.B411;
        out.bloopterms = native.Bloopterms;
        out.b1loop = native.B1loop;
        out.btotal = native.Btotal;
        out.dbdfnl_local_tree = native.dBdfNL_local_tree;
        out.dbdfnl_local_B222 = native.dBdfNL_local_B222;
        out.dbdfnl_local_B321I = native.dBdfNL_local_B321I;
        out.dbdfnl_local_B321II = native.dBdfNL_local_B321II;
        out.dbdfnl_local_B411 = native.dBdfNL_local_B411;
        out.dbdfnl_local_1loop = native.dBdfNL_local_1loop;
        out.dbdfnl_local_total = native.dBdfNL_local_total;
        out.dbdfnl_local_primordial = native.dBdfNL_local_primordial;
        out.dbdfnl_local_bphi_f2 = native.dBdfNL_local_bphi_f2;
        out.dbdfnl_local_bphi_advection = native.dBdfNL_local_bphi_advection;
        out.dbdfnl_local_bphidelta = native.dBdfNL_local_bphidelta;
        out.dbdfnl_local_bphi_b2 = native.dBdfNL_local_bphi_b2;
        out.dbdfnl_local_bphi_bK2 = native.dBdfNL_local_bphi_bK2;
        out.dbdfnl_local_bphi_reconstruction = native.dBdfNL_local_bphi_reconstruction;
        out.bhalo_tree_fnl2_bphi_B0 =
            native.Bhalo_tree_fNL2_bphi_B0;
        out.bhalo_tree_fnl2_bphi_sq_advection =
            native.Bhalo_tree_fNL2_bphi_sq_advection;
        out.bhalo_tree_fnl2_bphi_sq_F2 =
            native.Bhalo_tree_fNL2_bphi_sq_F2;
        out.bhalo_tree_fnl2_bphi_sq_b2 =
            native.Bhalo_tree_fNL2_bphi_sq_b2;
        out.bhalo_tree_fnl2_bphi_sq_bK2 =
            native.Bhalo_tree_fNL2_bphi_sq_bK2;
        out.bhalo_tree_fnl2_bphi_sq_reconstruction =
            native.Bhalo_tree_fNL2_bphi_sq_reconstruction;
        out.bhalo_tree_fnl2_bphi_bphidelta =
            native.Bhalo_tree_fNL2_bphi_bphidelta;
        out.bhalo_tree_fnl2_bphi2_operator =
            native.Bhalo_tree_fNL2_bphi2_operator;
        out.bhalo_tree_fnl2_deterministic =
            native.Bhalo_tree_fNL2_deterministic;
        out.bhalo_tree_fnl2_stochastic_alpha3PNG_basis =
            native.Bhalo_tree_fNL2_stochastic_alpha3PNG_basis;
        out.stochastic_alpha3_basis = native.stochastic_alpha3_basis;
        out.stochastic_alpha4_basis = native.stochastic_alpha4_basis;
        out.dbdfnl_stochastic_alpha3_basis = native.dBdfNL_stochastic_alpha3_basis;

        out.b222_abserr = native.B222_stats.abserr;
        out.b321i_abserr = native.B321I_stats.abserr;
        out.b411_abserr = native.B411_stats.abserr;
        out.p13_k1_abserr = native.P13_k1_stats.abserr;
        out.p13_k2_abserr = native.P13_k2_stats.abserr;
        out.p13_k3_abserr = native.P13_k3_stats.abserr;
        out.dbdfnl_local_B222_abserr = native.dBdfNL_local_B222_stats.abserr;
        out.dbdfnl_local_B321I_abserr = native.dBdfNL_local_B321I_stats.abserr;
        out.dbdfnl_local_B321II_abserr = native.dBdfNL_local_B321II_stats.abserr;
        out.dbdfnl_local_B411_abserr = native.dBdfNL_local_B411_stats.abserr;
        out.dp13dfnl_local_k1_abserr = native.dP13dfNL_local_k1_stats.abserr;
        out.dp13dfnl_local_k2_abserr = native.dP13dfNL_local_k2_stats.abserr;
        out.dp13dfnl_local_k3_abserr = native.dP13dfNL_local_k3_stats.abserr;
        out.b222_neval = native.B222_stats.neval;
        out.b321i_neval = native.B321I_stats.neval;
        out.b411_neval = native.B411_stats.neval;
        out.p13_k1_neval = native.P13_k1_stats.neval;
        out.p13_k2_neval = native.P13_k2_stats.neval;
        out.p13_k3_neval = native.P13_k3_stats.neval;
        out.dbdfnl_local_B222_neval = native.dBdfNL_local_B222_stats.neval;
        out.dbdfnl_local_B321I_neval = native.dBdfNL_local_B321I_stats.neval;
        out.dbdfnl_local_B321II_neval = native.dBdfNL_local_B321II_stats.neval;
        out.dbdfnl_local_B411_neval = native.dBdfNL_local_B411_stats.neval;
        out.dp13dfnl_local_k1_neval = native.dP13dfNL_local_k1_stats.neval;
        out.dp13dfnl_local_k2_neval = native.dP13dfNL_local_k2_stats.neval;
        out.dp13dfnl_local_k3_neval = native.dP13dfNL_local_k3_stats.neval;
        return out;
    }

  private:
    const PowerSpectrum& p_l_;
    const PowerSpectrum& transfer_m_;
    bool post_recon_ = false;
    marisa_b_native::NativeConfig config_;
    marisa_b_native::HaloBiasV1Params bias_;
};

class NativePostReconIRNoWiggleBackend : public BispectrumBackend {
  public:
    NativePostReconIRNoWiggleBackend(const PowerSpectrum& p_l, const PowerSpectrum& p_nw, const Options& opts)
        : p_l_(p_l), p_nw_(p_nw) {
        config_.qmin = opts.qmin;
        config_.qmax = opts.qmax;
        config_.epsrel = opts.epsrel;
        config_.p13_epsrel = opts.p13_epsrel;
        config_.smoothing_radius = opts.smoothing_radius;
        config_.bias_recon = opts.bias_recon;
        config_.recon_cellsize = opts.recon_cellsize;
        config_.recon_cic_window_power = opts.recon_cic_window_power;
        config_.singular_floor = opts.singular_floor;
        config_.ir_sigma2 = opts.ir_sigma2;
    }

    ComponentResult compute(const Triangle& tri) const override {
        marisa_b_native::Triangle native_tri;
        native_tri.k1 = tri.k1;
        native_tri.k2 = tri.k2;
        native_tri.mu12 = tri.mu12;
        const marisa_b_native::ComponentResult native =
            marisa_b_native::compute_post_recon_gaussian_ir_nowiggle(p_l_, p_nw_, native_tri, config_);

        ComponentResult out;
        out.triangle = tri;
        out.k3 = native.k3;
        out.btree = native.Btree;
        out.b222 = native.B222;
        out.b321i = native.B321I;
        out.b321ii = native.B321II;
        out.b411 = native.B411;
        out.bloopterms = native.Bloopterms;
        out.b1loop = native.B1loop;
        out.btotal = native.Btotal;
        out.b222_abserr = native.B222_stats.abserr;
        out.b321i_abserr = native.B321I_stats.abserr;
        out.b411_abserr = native.B411_stats.abserr;
        out.p13_k1_abserr = native.P13_k1_stats.abserr;
        out.p13_k2_abserr = native.P13_k2_stats.abserr;
        out.p13_k3_abserr = native.P13_k3_stats.abserr;
        out.b222_neval = native.B222_stats.neval;
        out.b321i_neval = native.B321I_stats.neval;
        out.b411_neval = native.B411_stats.neval;
        out.p13_k1_neval = native.P13_k1_stats.neval;
        out.p13_k2_neval = native.P13_k2_stats.neval;
        out.p13_k3_neval = native.P13_k3_stats.neval;
        return out;
    }

  private:
    const PowerSpectrum& p_l_;
    const PowerSpectrum& p_nw_;
    marisa_b_native::NativeConfig config_;
};

class NativePngTreeBackend : public BispectrumBackend {
  public:
    NativePngTreeBackend(const PowerSpectrum& p_l, const PowerSpectrum& transfer_m, double b1)
        : p_l_(p_l), transfer_m_(transfer_m), b1_(b1) {}

    ComponentResult compute(const Triangle& tri) const override {
        marisa_b_native::Triangle native_tri;
        native_tri.k1 = tri.k1;
        native_tri.k2 = tri.k2;
        native_tri.mu12 = tri.mu12;

        ComponentResult out;
        out.triangle = tri;
        out.k3 = marisa_b_native::triangle_k3(native_tri);

        const double f2a = marisa_b_native::F2eds(tri.k1, tri.k2, tri.mu12);
        const double f2b = marisa_b_native::F2eds(tri.k1, out.k3, -(tri.k2 * tri.mu12 + tri.k1) / out.k3);
        const double f2c = marisa_b_native::F2eds(tri.k2, out.k3, -(tri.k1 * tri.mu12 + tri.k2) / out.k3);
        out.btree = 2.0 * (p_l_(tri.k1) * p_l_(tri.k2) * f2a
                            + p_l_(out.k3) * p_l_(tri.k1) * f2b
                            + p_l_(tri.k2) * p_l_(out.k3) * f2c);
        out.dbdfnl_local_tree = marisa_b_native::local_png_tree_dfNL(p_l_, transfer_m_, native_tri, b1_);
        return out;
    }

  private:
    const PowerSpectrum& p_l_;
    const PowerSpectrum& transfer_m_;
    double b1_ = 1.0;
};

class NativePngOneLoopBackend : public BispectrumBackend {
  public:
    NativePngOneLoopBackend(const PowerSpectrum& p_l, const PowerSpectrum& transfer_m, const Options& opts)
        : p_l_(p_l), transfer_m_(transfer_m), b1_(opts.b1) {
        config_.qmin = opts.qmin;
        config_.qmax = opts.qmax;
        config_.epsrel = opts.epsrel;
        config_.p13_epsrel = opts.p13_epsrel;
        config_.singular_floor = opts.singular_floor;
    }

    ComponentResult compute(const Triangle& tri) const override {
        marisa_b_native::Triangle native_tri;
        native_tri.k1 = tri.k1;
        native_tri.k2 = tri.k2;
        native_tri.mu12 = tri.mu12;
        const marisa_b_native::ComponentResult native =
            marisa_b_native::compute_pre_recon_local_png_1loop_dfNL(p_l_, transfer_m_, native_tri, config_, b1_);

        ComponentResult out;
        out.triangle = tri;
        out.k3 = native.k3;
        out.btree = native.Btree;
        out.b222 = native.B222;
        out.b321i = native.B321I;
        out.b321ii = native.B321II;
        out.b411 = native.B411;
        out.bloopterms = native.Bloopterms;
        out.b1loop = native.B1loop;
        out.btotal = native.Btotal;
        out.dbdfnl_local_tree = native.dBdfNL_local_tree;
        out.dbdfnl_local_B122I = native.dBdfNL_local_B122I;
        out.dbdfnl_local_B122II = native.dBdfNL_local_B122II;
        out.dbdfnl_local_B113I = native.dBdfNL_local_B113I;
        out.dbdfnl_local_B113II = native.dBdfNL_local_B113II;
        out.dbdfnl_local_1loop = native.dBdfNL_local_1loop;
        out.dbdfnl_local_total = native.dBdfNL_local_total;
        out.b222_abserr = native.B222_stats.abserr;
        out.b321i_abserr = native.B321I_stats.abserr;
        out.b411_abserr = native.B411_stats.abserr;
        out.p13_k1_abserr = native.P13_k1_stats.abserr;
        out.p13_k2_abserr = native.P13_k2_stats.abserr;
        out.p13_k3_abserr = native.P13_k3_stats.abserr;
        out.p12_png_k1_abserr = native.P12_png_k1_stats.abserr;
        out.p12_png_k2_abserr = native.P12_png_k2_stats.abserr;
        out.p12_png_k3_abserr = native.P12_png_k3_stats.abserr;
        out.dbdfnl_local_B122II_abserr = native.dBdfNL_local_B122II_stats.abserr;
        out.dbdfnl_local_B113II_abserr = native.dBdfNL_local_B113II_stats.abserr;
        out.b222_neval = native.B222_stats.neval;
        out.b321i_neval = native.B321I_stats.neval;
        out.b411_neval = native.B411_stats.neval;
        out.p13_k1_neval = native.P13_k1_stats.neval;
        out.p13_k2_neval = native.P13_k2_stats.neval;
        out.p13_k3_neval = native.P13_k3_stats.neval;
        out.p12_png_k1_neval = native.P12_png_k1_stats.neval;
        out.p12_png_k2_neval = native.P12_png_k2_stats.neval;
        out.p12_png_k3_neval = native.P12_png_k3_stats.neval;
        out.dbdfnl_local_B122II_neval = native.dBdfNL_local_B122II_stats.neval;
        out.dbdfnl_local_B113II_neval = native.dBdfNL_local_B113II_stats.neval;
        return out;
    }

  private:
    const PowerSpectrum& p_l_;
    const PowerSpectrum& transfer_m_;
    marisa_b_native::NativeConfig config_;
    double b1_ = 1.0;
};

class NativePostReconPngOneLoopBackend : public BispectrumBackend {
  public:
    NativePostReconPngOneLoopBackend(const PowerSpectrum& p_l, const PowerSpectrum& transfer_m, const Options& opts)
        : p_l_(p_l), transfer_m_(transfer_m), b1_(opts.b1) {
        config_.qmin = opts.qmin;
        config_.qmax = opts.qmax;
        config_.epsrel = opts.epsrel;
        config_.p13_epsrel = opts.p13_epsrel;
        config_.smoothing_radius = opts.smoothing_radius;
        config_.bias_recon = opts.bias_recon;
        config_.recon_cellsize = opts.recon_cellsize;
        config_.recon_cic_window_power = opts.recon_cic_window_power;
        config_.singular_floor = opts.singular_floor;
    }

    ComponentResult compute(const Triangle& tri) const override {
        marisa_b_native::Triangle native_tri;
        native_tri.k1 = tri.k1;
        native_tri.k2 = tri.k2;
        native_tri.mu12 = tri.mu12;
        const marisa_b_native::ComponentResult native =
            marisa_b_native::compute_post_recon_local_png_1loop_dfNL(p_l_, transfer_m_, native_tri, config_, b1_);

        ComponentResult out;
        out.triangle = tri;
        out.k3 = native.k3;
        out.btree = native.Btree;
        out.b222 = native.B222;
        out.b321i = native.B321I;
        out.b321ii = native.B321II;
        out.b411 = native.B411;
        out.bloopterms = native.Bloopterms;
        out.b1loop = native.B1loop;
        out.btotal = native.Btotal;
        out.dbdfnl_local_tree = native.dBdfNL_local_tree;
        out.dbdfnl_local_B122I = native.dBdfNL_local_B122I;
        out.dbdfnl_local_B122II = native.dBdfNL_local_B122II;
        out.dbdfnl_local_B113I = native.dBdfNL_local_B113I;
        out.dbdfnl_local_B113II = native.dBdfNL_local_B113II;
        out.dbdfnl_local_1loop = native.dBdfNL_local_1loop;
        out.dbdfnl_local_total = native.dBdfNL_local_total;
        out.p13_k1_abserr = native.P13_k1_stats.abserr;
        out.p13_k2_abserr = native.P13_k2_stats.abserr;
        out.p13_k3_abserr = native.P13_k3_stats.abserr;
        out.p12_png_k1_abserr = native.P12_png_k1_stats.abserr;
        out.p12_png_k2_abserr = native.P12_png_k2_stats.abserr;
        out.p12_png_k3_abserr = native.P12_png_k3_stats.abserr;
        out.dbdfnl_local_B122II_abserr = native.dBdfNL_local_B122II_stats.abserr;
        out.dbdfnl_local_B113II_abserr = native.dBdfNL_local_B113II_stats.abserr;
        out.p13_k1_neval = native.P13_k1_stats.neval;
        out.p13_k2_neval = native.P13_k2_stats.neval;
        out.p13_k3_neval = native.P13_k3_stats.neval;
        out.p12_png_k1_neval = native.P12_png_k1_stats.neval;
        out.p12_png_k2_neval = native.P12_png_k2_stats.neval;
        out.p12_png_k3_neval = native.P12_png_k3_stats.neval;
        out.dbdfnl_local_B122II_neval = native.dBdfNL_local_B122II_stats.neval;
        out.dbdfnl_local_B113II_neval = native.dBdfNL_local_B113II_stats.neval;
        return out;
    }

  private:
    const PowerSpectrum& p_l_;
    const PowerSpectrum& transfer_m_;
    marisa_b_native::NativeConfig config_;
    double b1_ = 1.0;
};

/*
 * single-source local-PNG B112II/fNL^2 的隔离诊断 backend。
 *
 * 该类只填写 b112ii_fnl2_coefficient 及其积分元数据；不会改变 Gaussian
 * Btotal 或线性 dB/dfNL 的生产语义。post_recon 仅决定 F2/Z_rec,2 kernel。
 */
class NativePngB112IIDiagnosticBackend : public BispectrumBackend {
  public:
    NativePngB112IIDiagnosticBackend(
        const PowerSpectrum& p_l,
        const PowerSpectrum& transfer_m,
        const Options& opts,
        bool post_recon)
        : p_l_(p_l), transfer_m_(transfer_m), post_recon_(post_recon) {
        config_.qmin = opts.qmin;
        config_.qmax = opts.qmax;
        config_.png_ir_cutoff = opts.png_ir_cutoff;
        config_.png_b112ii_multicenter_qmc =
            opts.b112ii_integrator == "multicenter_qmc";
        config_.png_b112ii_qmc_power = opts.b112ii_qmc_power;
        config_.png_b112ii_qmc_replicates = opts.b112ii_qmc_replicates;
        config_.epsrel = opts.epsrel;
        config_.p13_epsrel = opts.p13_epsrel;
        config_.smoothing_radius = opts.smoothing_radius;
        config_.bias_recon = opts.bias_recon;
        config_.recon_cellsize = opts.recon_cellsize;
        config_.recon_cic_window_power = opts.recon_cic_window_power;
        config_.singular_floor = opts.singular_floor;
    }

    ComponentResult compute(const Triangle& tri) const override {
        marisa_b_native::Triangle native_tri;
        native_tri.k1 = tri.k1;
        native_tri.k2 = tri.k2;
        native_tri.mu12 = tri.mu12;
        const marisa_b_native::ComponentResult native = post_recon_
            ? marisa_b_native::compute_post_recon_local_png_B112II_fNL2_coefficient(
                  p_l_, transfer_m_, native_tri, config_)
            : marisa_b_native::compute_pre_recon_local_png_B112II_fNL2_coefficient(
                  p_l_, transfer_m_, native_tri, config_);

        ComponentResult out;
        out.triangle = tri;
        out.k3 = native.k3;
        out.b112ii_fnl2_coefficient = native.B112II_fNL2_coefficient;
        out.b112ii_fnl2_coefficient_abserr =
            native.B112II_fNL2_coefficient_stats.abserr;
        out.b112ii_fnl2_coefficient_neval =
            native.B112II_fNL2_coefficient_stats.neval;
        return out;
    }

  private:
    const PowerSpectrum& p_l_;
    const PowerSpectrum& transfer_m_;
    bool post_recon_ = false;
    marisa_b_native::NativeConfig config_;
};

/*
 * 通过隔离 gate 后启用的 finite-fNL DM backend。
 *
 * Btree/B1loop 仍是 Gaussian 基线；Btotal 才是
 * B_G + fNL*R1 + fNL^2*R2。旧 response-only 与 diagnostic mode 完全不走
 * 该类，因此不会改变现有 profiler 输入语义。
 */
class NativePngFiniteOneLoopBackend : public BispectrumBackend {
  public:
    NativePngFiniteOneLoopBackend(
        const PowerSpectrum& p_l,
        const PowerSpectrum& transfer_m,
        const Options& opts,
        bool post_recon)
        : p_l_(p_l),
          transfer_m_(transfer_m),
          post_recon_(post_recon),
          fnl_(opts.fnl) {
        config_.qmin = opts.qmin;
        config_.qmax = opts.qmax;
        config_.png_ir_cutoff = opts.png_ir_cutoff;
        config_.png_b112ii_multicenter_qmc =
            opts.b112ii_integrator == "multicenter_qmc";
        config_.png_b112ii_qmc_power = opts.b112ii_qmc_power;
        config_.png_b112ii_qmc_replicates = opts.b112ii_qmc_replicates;
        config_.epsrel = opts.epsrel;
        config_.p13_epsrel = opts.p13_epsrel;
        config_.smoothing_radius = opts.smoothing_radius;
        config_.bias_recon = opts.bias_recon;
        config_.recon_cellsize = opts.recon_cellsize;
        config_.recon_cic_window_power = opts.recon_cic_window_power;
        config_.singular_floor = opts.singular_floor;
    }

    ComponentResult compute(const Triangle& tri) const override {
        marisa_b_native::Triangle native_tri;
        native_tri.k1 = tri.k1;
        native_tri.k2 = tri.k2;
        native_tri.mu12 = tri.mu12;
        const marisa_b_native::ComponentResult native = post_recon_
            ? marisa_b_native::compute_post_recon_local_png_finite_1loop(
                  p_l_, transfer_m_, native_tri, config_, fnl_)
            : marisa_b_native::compute_pre_recon_local_png_finite_1loop(
                  p_l_, transfer_m_, native_tri, config_, fnl_);

        ComponentResult out;
        out.triangle = tri;
        out.k3 = native.k3;
        out.btree = native.Btree;
        out.b222 = native.B222;
        out.b321i = native.B321I;
        out.b321ii = native.B321II;
        out.b411 = native.B411;
        out.bloopterms = native.Bloopterms;
        out.b1loop = native.B1loop;
        out.btotal = native.Btotal;
        out.dbdfnl_local_tree = native.dBdfNL_local_tree;
        out.dbdfnl_local_B122I = native.dBdfNL_local_B122I;
        out.dbdfnl_local_B122II = native.dBdfNL_local_B122II;
        out.dbdfnl_local_B113I = native.dBdfNL_local_B113I;
        out.dbdfnl_local_B113II = native.dBdfNL_local_B113II;
        out.dbdfnl_local_1loop = native.dBdfNL_local_1loop;
        out.dbdfnl_local_total = native.dBdfNL_local_total;
        out.b112ii_fnl2_coefficient =
            native.B112II_fNL2_coefficient;

        out.b222_abserr = native.B222_stats.abserr;
        out.b321i_abserr = native.B321I_stats.abserr;
        out.b411_abserr = native.B411_stats.abserr;
        out.p13_k1_abserr = native.P13_k1_stats.abserr;
        out.p13_k2_abserr = native.P13_k2_stats.abserr;
        out.p13_k3_abserr = native.P13_k3_stats.abserr;
        out.p12_png_k1_abserr = native.P12_png_k1_stats.abserr;
        out.p12_png_k2_abserr = native.P12_png_k2_stats.abserr;
        out.p12_png_k3_abserr = native.P12_png_k3_stats.abserr;
        out.dbdfnl_local_B122II_abserr =
            native.dBdfNL_local_B122II_stats.abserr;
        out.dbdfnl_local_B113II_abserr =
            native.dBdfNL_local_B113II_stats.abserr;
        out.b112ii_fnl2_coefficient_abserr =
            native.B112II_fNL2_coefficient_stats.abserr;

        out.b222_neval = native.B222_stats.neval;
        out.b321i_neval = native.B321I_stats.neval;
        out.b411_neval = native.B411_stats.neval;
        out.p13_k1_neval = native.P13_k1_stats.neval;
        out.p13_k2_neval = native.P13_k2_stats.neval;
        out.p13_k3_neval = native.P13_k3_stats.neval;
        out.p12_png_k1_neval = native.P12_png_k1_stats.neval;
        out.p12_png_k2_neval = native.P12_png_k2_stats.neval;
        out.p12_png_k3_neval = native.P12_png_k3_stats.neval;
        out.dbdfnl_local_B122II_neval =
            native.dBdfNL_local_B122II_stats.neval;
        out.dbdfnl_local_B113II_neval =
            native.dBdfNL_local_B113II_stats.neval;
        out.b112ii_fnl2_coefficient_neval =
            native.B112II_fNL2_coefficient_stats.neval;
        return out;
    }

  private:
    const PowerSpectrum& p_l_;
    const PowerSpectrum& transfer_m_;
    bool post_recon_ = false;
    double fnl_ = 0.0;
    marisa_b_native::NativeConfig config_;
};

class MarisaBEngine {
  public:
    explicit MarisaBEngine(const BispectrumBackend& backend) : backend_(backend) {}

    std::vector<ComponentResult> compute_many(const std::vector<Triangle>& triangles) const {
        std::vector<ComponentResult> out;
        out.reserve(triangles.size());
        for (std::vector<Triangle>::const_iterator it = triangles.begin(); it != triangles.end(); ++it) {
            out.push_back(backend_.compute(*it));
        }
        return out;
    }

  private:
    const BispectrumBackend& backend_;
};

void print_json_string(const std::string& value) {
    std::cout << '"';
    for (std::string::const_iterator it = value.begin(); it != value.end(); ++it) {
        if (*it == '"' || *it == '\\') {
            std::cout << '\\' << *it;
        } else if (*it == '\n') {
            std::cout << "\\n";
        } else {
            std::cout << *it;
        }
    }
    std::cout << '"';
}

void print_component_json(const ComponentResult& row, int index, bool include_halo_bias_v1) {
    std::cout << "    {\n";
    std::cout << "      \"triangle_index\": " << index << ",\n";
    std::cout << "      \"k1\": " << row.triangle.k1 << ",\n";
    std::cout << "      \"k2\": " << row.triangle.k2 << ",\n";
    std::cout << "      \"mu12\": " << row.triangle.mu12 << ",\n";
    std::cout << "      \"k3\": " << row.k3 << ",\n";
    std::cout << "      \"Btree\": " << row.btree << ",\n";
    std::cout << "      \"B222\": " << row.b222 << ",\n";
    std::cout << "      \"B321I\": " << row.b321i << ",\n";
    std::cout << "      \"B321II\": " << row.b321ii << ",\n";
    std::cout << "      \"B411\": " << row.b411 << ",\n";
    std::cout << "      \"Bloopterms\": " << row.bloopterms << ",\n";
    std::cout << "      \"B1loop\": " << row.b1loop << ",\n";
    std::cout << "      \"Btotal\": " << row.btotal << ",\n";
    std::cout << "      \"dBdfNL_local_tree\": " << row.dbdfnl_local_tree << ",\n";
    std::cout << "      \"dBdfNL_local_B122I\": " << row.dbdfnl_local_B122I << ",\n";
    std::cout << "      \"dBdfNL_local_B122II\": " << row.dbdfnl_local_B122II << ",\n";
    std::cout << "      \"dBdfNL_local_B113I\": " << row.dbdfnl_local_B113I << ",\n";
    std::cout << "      \"dBdfNL_local_B113II\": " << row.dbdfnl_local_B113II << ",\n";
    std::cout << "      \"dBdfNL_local_1loop\": " << row.dbdfnl_local_1loop << ",\n";
    std::cout << "      \"dBdfNL_local_total\": " << row.dbdfnl_local_total << ",\n";
    std::cout << "      \"B112II_fNL2_coefficient\": " << row.b112ii_fnl2_coefficient << ",\n";
    if (include_halo_bias_v1) {
        std::cout << "      \"dBdfNL_local_B222\": " << row.dbdfnl_local_B222 << ",\n";
        std::cout << "      \"dBdfNL_local_B321I\": " << row.dbdfnl_local_B321I << ",\n";
        std::cout << "      \"dBdfNL_local_B321II\": " << row.dbdfnl_local_B321II << ",\n";
        std::cout << "      \"dBdfNL_local_B411\": " << row.dbdfnl_local_B411 << ",\n";
        std::cout << "      \"dBdfNL_local_primordial\": " << row.dbdfnl_local_primordial << ",\n";
        std::cout << "      \"dBdfNL_local_bphi_f2\": " << row.dbdfnl_local_bphi_f2 << ",\n";
        std::cout << "      \"dBdfNL_local_bphi_advection\": " << row.dbdfnl_local_bphi_advection << ",\n";
        std::cout << "      \"dBdfNL_local_bphidelta\": " << row.dbdfnl_local_bphidelta << ",\n";
        std::cout << "      \"dBdfNL_local_bphi_b2\": " << row.dbdfnl_local_bphi_b2 << ",\n";
        std::cout << "      \"dBdfNL_local_bphi_bK2\": " << row.dbdfnl_local_bphi_bK2 << ",\n";
        std::cout << "      \"dBdfNL_local_bphi_reconstruction\": " << row.dbdfnl_local_bphi_reconstruction << ",\n";
        std::cout << "      \"Bhalo_tree_fNL2_bphi_B0\": " << row.bhalo_tree_fnl2_bphi_B0 << ",\n";
        std::cout << "      \"Bhalo_tree_fNL2_bphi_sq_advection\": " << row.bhalo_tree_fnl2_bphi_sq_advection << ",\n";
        std::cout << "      \"Bhalo_tree_fNL2_bphi_sq_F2\": " << row.bhalo_tree_fnl2_bphi_sq_F2 << ",\n";
        std::cout << "      \"Bhalo_tree_fNL2_bphi_sq_b2\": " << row.bhalo_tree_fnl2_bphi_sq_b2 << ",\n";
        std::cout << "      \"Bhalo_tree_fNL2_bphi_sq_bK2\": " << row.bhalo_tree_fnl2_bphi_sq_bK2 << ",\n";
        std::cout << "      \"Bhalo_tree_fNL2_bphi_sq_reconstruction\": " << row.bhalo_tree_fnl2_bphi_sq_reconstruction << ",\n";
        std::cout << "      \"Bhalo_tree_fNL2_bphi_bphidelta\": " << row.bhalo_tree_fnl2_bphi_bphidelta << ",\n";
        std::cout << "      \"Bhalo_tree_fNL2_bphi2_operator\": " << row.bhalo_tree_fnl2_bphi2_operator << ",\n";
        std::cout << "      \"Bhalo_tree_fNL2_deterministic\": " << row.bhalo_tree_fnl2_deterministic << ",\n";
        std::cout << "      \"Bhalo_tree_fNL2_stochastic_alpha3PNG_basis\": " << row.bhalo_tree_fnl2_stochastic_alpha3PNG_basis << ",\n";
        std::cout << "      \"stochastic_alpha3_basis\": " << row.stochastic_alpha3_basis << ",\n";
        std::cout << "      \"stochastic_alpha4_basis\": " << row.stochastic_alpha4_basis << ",\n";
        std::cout << "      \"dBdfNL_stochastic_alpha3_basis\": " << row.dbdfnl_stochastic_alpha3_basis << ",\n";
    }
    std::cout << "      \"integration_metadata\": {\n";
    std::cout << "        \"B222\": {\"abserr\": " << row.b222_abserr << ", \"neval\": " << row.b222_neval << "},\n";
    std::cout << "        \"B321I\": {\"abserr\": " << row.b321i_abserr << ", \"neval\": " << row.b321i_neval << "},\n";
    std::cout << "        \"B411\": {\"abserr\": " << row.b411_abserr << ", \"neval\": " << row.b411_neval << "},\n";
    std::cout << "        \"P13_k1\": {\"abserr\": " << row.p13_k1_abserr << ", \"neval\": " << row.p13_k1_neval << "},\n";
    std::cout << "        \"P13_k2\": {\"abserr\": " << row.p13_k2_abserr << ", \"neval\": " << row.p13_k2_neval << "},\n";
    std::cout << "        \"P13_k3\": {\"abserr\": " << row.p13_k3_abserr << ", \"neval\": " << row.p13_k3_neval << "},\n";
    std::cout << "        \"P12_png_k1\": {\"abserr\": " << row.p12_png_k1_abserr << ", \"neval\": " << row.p12_png_k1_neval << "},\n";
    std::cout << "        \"P12_png_k2\": {\"abserr\": " << row.p12_png_k2_abserr << ", \"neval\": " << row.p12_png_k2_neval << "},\n";
    std::cout << "        \"P12_png_k3\": {\"abserr\": " << row.p12_png_k3_abserr << ", \"neval\": " << row.p12_png_k3_neval << "},\n";
    std::cout << "        \"dBdfNL_local_B122II\": {\"abserr\": " << row.dbdfnl_local_B122II_abserr << ", \"neval\": " << row.dbdfnl_local_B122II_neval << "},\n";
    std::cout << "        \"dBdfNL_local_B113II\": {\"abserr\": " << row.dbdfnl_local_B113II_abserr << ", \"neval\": " << row.dbdfnl_local_B113II_neval << "},\n";
    std::cout << "        \"B112II_fNL2_coefficient\": {\"abserr\": " << row.b112ii_fnl2_coefficient_abserr << ", \"neval\": " << row.b112ii_fnl2_coefficient_neval << "}";
    if (include_halo_bias_v1) {
        std::cout << ",\n";
        std::cout << "        \"dBdfNL_local_B222\": {\"abserr\": " << row.dbdfnl_local_B222_abserr << ", \"neval\": " << row.dbdfnl_local_B222_neval << "},\n";
        std::cout << "        \"dBdfNL_local_B321I\": {\"abserr\": " << row.dbdfnl_local_B321I_abserr << ", \"neval\": " << row.dbdfnl_local_B321I_neval << "},\n";
        std::cout << "        \"dBdfNL_local_B321II\": {\"abserr\": " << row.dbdfnl_local_B321II_abserr << ", \"neval\": " << row.dbdfnl_local_B321II_neval << "},\n";
        std::cout << "        \"dBdfNL_local_B411\": {\"abserr\": " << row.dbdfnl_local_B411_abserr << ", \"neval\": " << row.dbdfnl_local_B411_neval << "},\n";
        std::cout << "        \"dP13dfNL_local_k1\": {\"abserr\": " << row.dp13dfnl_local_k1_abserr << ", \"neval\": " << row.dp13dfnl_local_k1_neval << "},\n";
        std::cout << "        \"dP13dfNL_local_k2\": {\"abserr\": " << row.dp13dfnl_local_k2_abserr << ", \"neval\": " << row.dp13dfnl_local_k2_neval << "},\n";
        std::cout << "        \"dP13dfNL_local_k3\": {\"abserr\": " << row.dp13dfnl_local_k3_abserr << ", \"neval\": " << row.dp13dfnl_local_k3_neval << "}\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "      }\n";
    std::cout << "    }";
}

void print_payload_json(const Options& opts, const std::vector<ComponentResult>& rows, double wall_seconds) {
    const bool finite_png =
        opts.mode == "pre_recon_local_png_finite_1loop"
        || opts.mode == "post_recon_local_png_finite_1loop";
    const bool halo_png_one_loop_truncated =
        opts.mode == "pre_recon_halo_bias_v1_local_png_1loop_truncated"
        || opts.mode == "post_recon_halo_bias_v1_local_png_1loop_truncated";
    const bool halo_bias_v1 =
        opts.mode == "pre_recon_halo_bias_v1_gaussian"
        || opts.mode == "post_recon_halo_bias_v1_gaussian"
        || opts.mode == "pre_recon_halo_bias_v1_local_png_tree"
        || opts.mode == "post_recon_halo_bias_v1_local_png_tree"
        || halo_png_one_loop_truncated;
    std::cout << std::setprecision(17);
    std::cout << "{\n";
    std::cout << "  \"program\": \"MARISA-B\",\n";
    std::cout << "  \"version\": ";
    print_json_string(halo_bias_v1 ? "v1-halo-bias" : "v0-native-core");
    std::cout << ",\n";
    std::cout << "  \"mode\": ";
    print_json_string(opts.mode);
    std::cout << ",\n";
    std::cout << "  \"backend\": ";
    print_json_string(opts.backend);
    std::cout << ",\n";
    std::cout << "  \"pk_table\": ";
    print_json_string(opts.pk_table);
    std::cout << ",\n";
    std::cout << "  \"png_table\": ";
    print_json_string(opts.png_table);
    std::cout << ",\n";
    std::cout << "  \"nowiggle_table\": ";
    print_json_string(opts.nowiggle_table);
    std::cout << ",\n";
    std::cout << "  \"epsrel\": " << opts.epsrel << ",\n";
    std::cout << "  \"p13_epsrel\": " << opts.p13_epsrel << ",\n";
    std::cout << "  \"qmin\": " << opts.qmin << ",\n";
    std::cout << "  \"qmax\": " << opts.qmax << ",\n";
    std::cout << "  \"png_ir_cutoff\": " << opts.png_ir_cutoff << ",\n";
    std::cout << "  \"b112ii_integrator\": ";
    print_json_string(opts.b112ii_integrator);
    std::cout << ",\n";
    std::cout << "  \"b112ii_qmc_power\": " << opts.b112ii_qmc_power << ",\n";
    std::cout << "  \"b112ii_qmc_replicates\": "
              << opts.b112ii_qmc_replicates << ",\n";
    std::cout << "  \"smoothing_radius\": " << opts.smoothing_radius << ",\n";
    std::cout << "  \"bias_recon\": " << opts.bias_recon << ",\n";
    std::cout << "  \"recon_cellsize\": " << opts.recon_cellsize << ",\n";
    std::cout << "  \"recon_cic_window_power\": " << opts.recon_cic_window_power << ",\n";
    std::cout << "  \"singular_floor\": " << opts.singular_floor << ",\n";
    std::cout << "  \"ir_sigma2\": " << opts.ir_sigma2 << ",\n";
    std::cout << "  \"b1\": " << opts.b1 << ",\n";
    if (finite_png) {
        std::cout << "  \"fNL\": " << opts.fnl << ",\n";
    }
    if (halo_bias_v1) {
        std::cout << "  \"b2\": " << opts.b2 << ",\n";
        std::cout << "  \"bK2\": " << opts.bK2 << ",\n";
        if (opts.mode == "pre_recon_halo_bias_v1_local_png_tree" || opts.mode == "post_recon_halo_bias_v1_local_png_tree" || halo_png_one_loop_truncated) {
            std::cout << "  \"bphi\": " << opts.bphi << ",\n";
            std::cout << "  \"bphidelta\": " << opts.bphidelta << ",\n";
            std::cout << "  \"bphi2\": " << opts.bphi2 << ",\n";
        }
    }
    std::cout << "  \"growth_mode\": \"unity_input_pk_at_target_z\",\n";
    std::cout << "  \"metadata\": {\n";
    std::cout << "    \"wall_seconds\": " << wall_seconds << ",\n";
    if (halo_png_one_loop_truncated) {
        std::cout << "    \"model_name\": \"MARISA-B halo bias-v1 truncated fixed-cutoff one-loop local-PNG response\",\n";
    } else if (opts.mode == "pre_recon_halo_bias_v1_gaussian"
            || opts.mode == "post_recon_halo_bias_v1_gaussian") {
        std::cout << "    \"model_name\": \"MARISA-B halo bias-v1 truncated fixed-cutoff one-loop Gaussian\",\n";
    }
    std::cout << "    \"component_source\": ";
    if (halo_png_one_loop_truncated) {
        print_json_string(
            opts.mode == "pre_recon_halo_bias_v1_local_png_1loop_truncated"
                ? "MARISA-B native C++ pre-recon halo bias-v1 local-PNG truncated one-loop response"
                : "MARISA-B native C++ post-recon halo bias-v1 local-PNG truncated one-loop response");
        std::cout << ",\n";
        std::cout << "    \"gaussian_bias_basis\": \"frozen_b1_b2_bK2_truncated\",\n";
        std::cout << "    \"png_bias_basis\": \"advected_bphi_bphidelta_truncated\",\n";
        std::cout << "    \"local_ic_order\": \"linear_fNL_single_source\",\n";
        std::cout << "    \"loop_order\": \"one_loop_spt_response\",\n";
        std::cout << "    \"renormalization\": \"fixed_cutoff_unrenormalized\",\n";
        std::cout << "    \"tree_reference\": \"Barreira_2022_exact\",\n";
        std::cout << "    \"loop_momentum_routing\": \"canonical_sorted_external_legs_native_gaussian_diagram_logq_with_spherical_qmin_qmax\",\n";
        std::cout << "    \"radial_integration_coordinate\": \"log(q/k1_canonical)\",\n";
        std::cout << "    \"note\": \"Fixed-cutoff, unrenormalized continuation of the Gaussian MARISA-B tree+B222+B321I+B321II+B411 functional. Before each new halo-v1 loop evaluation the three external side lengths are sorted and the two shortest define the native diagram q routing, making the regulated functional invariant under external-leg relabelling. Its radial cubature coordinate is log(q/k1_canonical), with the exact Jacobian and the same spherical qmin/qmax domain; legacy DM and explicit-PNG modes retain their historical linear coordinate. The PNG K1..K4 response is defined by local-IC quadratic substitution plus the advected phi(q) and phi(q)delta recurrences; reconstruction uses the exact linear product rule at fixed b_rec. Only its K1/K2 tree limit is the Barreira-2022 result, so this mode is not claimed as a strict halo-PNG one-loop derivation from that paper. The old explicit primordial-B0 loop path is not added, preventing double counting. At finite hard cutoff, momentum shifts relating this routing to the legacy explicit-B0 routing change the regulated integration domain; their residual difference is a routing/cutoff diagnostic, not a response-kernel coefficient adjustment.\"\n";
    } else if (opts.mode == "pre_recon_halo_bias_v1_local_png_tree" || opts.mode == "post_recon_halo_bias_v1_local_png_tree") {
        print_json_string(
            opts.mode == "pre_recon_halo_bias_v1_local_png_tree"
                ? "Barreira-2022 exact pre-recon halo local-PNG tree response"
                : "Barreira-2022 halo local-PNG tree response propagated through finite-R reconstruction");
        std::cout << ",\n";
        std::cout << "    \"note\": \"The linear response implements Eq. (4.2) and Appendix B.5/B.6 of Barreira 2022. Both pre and post modes export the complete fixed-bias deterministic tree-level fNL^2 coefficient of Dizgah et al. 2020 Eq. (2.58), including the independent b_phi2 operator and the exact finite-R quadratic reconstruction product-rule term, plus the raw leading quadratic PNG stochastic basis of Eq. (2.65). No halo-PNG K3/K4 or one-loop completion is claimed.\"\n";
    } else if (halo_bias_v1) {
        print_json_string(
            opts.mode == "pre_recon_halo_bias_v1_gaussian"
                ? "MARISA-B native C++ pre-recon halo bias-v1 Gaussian kernels"
                : "MARISA-B native C++ post-recon halo bias-v1 Gaussian kernels");
        std::cout << ",\n";
        std::cout << "    \"gaussian_bias_basis\": \"frozen_b1_b2_bK2_truncated\",\n";
        std::cout << "    \"loop_order\": \"one_loop_spt\",\n";
        std::cout << "    \"renormalization\": \"fixed_cutoff_unrenormalized\",\n";
        std::cout << "    \"loop_momentum_routing\": \"canonical_sorted_external_legs_native_gaussian_diagram_logq_with_spherical_qmin_qmax\",\n";
        std::cout << "    \"radial_integration_coordinate\": \"log(q/k1_canonical)\",\n";
        std::cout << "    \"note\": \"K_n=b1 F_n+b2 D_n+bK2 T_n for n<=4. Before each new halo-v1 loop evaluation the three external side lengths are sorted and the two shortest define the native diagram q routing, making this regulated functional invariant under external-leg relabelling. The new halo-v1 radial cubature uses log(q/k1_canonical) with the exact Jacobian and unchanged spherical qmin/qmax domain. The operator set follows Barreira 2022 Eq. (4.2), while K3/K4 are this project's mechanical perturbative continuation for MARISA-B one-loop and are not claimed as a one-loop derivation from that paper. Stochastic terms are profiled in the fit layer, not included here.\"\n";
    } else if (finite_png) {
        print_json_string(
            opts.mode == "pre_recon_local_png_finite_1loop"
                ? "MARISA-B pre-recon finite-fNL local-PNG DM one-loop model"
                : "MARISA-B post-recon finite-fNL local-PNG DM one-loop model");
        std::cout << ",\n";
        std::cout << "    \"local_png_order\": \"single_source_gNL0_fixed_PL_strict_O_PL3\",\n";
        std::cout << "    \"model_formula\": \"B_G_tree+B_G_1loop+fNL*R1+fNL^2*B112II_hat\",\n";
        std::cout << "    \"production_total_modified\": true,\n";
        std::cout << "    \"legacy_mode_semantics_modified\": false,\n";
        std::cout << "    \"note\": \"This explicit finite-fNL mode was enabled only after the isolated T0 permutation, independent-reference, IR/qmax, shell-projection, and pre/post-limit gates passed. Btree and B1loop remain Gaussian; dBdfNL_local_total remains R1; B112II_fNL2_coefficient remains R2; only this mode combines them into Btotal at the supplied fNL.\"\n";
    } else if (opts.mode == "pre_recon_local_png_b112ii_diagnostic"
               || opts.mode == "post_recon_local_png_b112ii_diagnostic") {
        print_json_string(
            opts.mode == "pre_recon_local_png_b112ii_diagnostic"
                ? "MARISA-B isolated pre-recon local-PNG B112II/fNL^2 diagnostic"
                : "MARISA-B isolated post-recon local-PNG B112II/fNL^2 diagnostic");
        std::cout << ",\n";
        std::cout << "    \"local_png_order\": \"single_source_fNL_squared_O_PL3\",\n";
        std::cout << "    \"production_total_modified\": false,\n";
        std::cout << "    \"note\": \"Computes the coefficient B112II/fNL^2 from the local primordial trispectrum with all twelve exchange terms and three B112 cyclic orientations. Every primordial P_phi line uses png_ir_cutoff (or qmin when zero). The coefficient is isolated and is not added to Btotal or dBdfNL.\"\n";
    } else if (opts.mode == "pre_recon_local_png_tree") {
        print_json_string("MARISA-B native C++ local PNG tree kernel using tabulated P_L and M=sqrt(P_L/P_phi)");
        std::cout << ",\n";
        std::cout << "    \"note\": \"dBdfNL_local_tree = 2*b1^3*[P1*P2*M3/(M1*M2)+cyc.]; no loop integration is performed.\"\n";
    } else if (opts.mode == "pre_recon_local_png_1loop") {
        print_json_string("MARISA-B native C++ local PNG tree plus linear-in-fNL one-loop kernels using tabulated P_L and M=sqrt(P_L/P_phi)");
        std::cout << ",\n";
        std::cout << "    \"note\": \"dBdfNL_local_total = B111 + B122I + B122II + B113I + B113II; B112II is fNL^2 and has zero derivative at fNL=0.\"\n";
    } else if (opts.mode == "post_recon_local_png_1loop") {
        print_json_string("MARISA-B native C++ post-recon local PNG linear response with finite-R Z_rec^(n) kernels");
        std::cout << ",\n";
        std::cout << "    \"note\": \"dBdfNL_local_total = B111 + B122I + B122II + B113I + B113II using Z_rec kernels; B112II is fNL^2 and has zero derivative at fNL=0.\"\n";
    } else if (opts.mode == "post_recon_gaussian") {
        print_json_string("MARISA-B native C++ post-recon Gaussian kernels with finite-R Z_rec^(n)");
        std::cout << ",\n";
        std::cout << "    \"note\": \"Schema compatibility: B411 stores post-recon B114; B321II stores post-recon B123I propagator correction.\"\n";
    } else if (opts.mode == "post_recon_gaussian_ir_nowiggle") {
        print_json_string("MARISA-B native C++ post-recon Gaussian 2403 Eq.51 IR/no-wiggle diagnostic");
        std::cout << ",\n";
        std::cout << "    \"note\": \"Btotal evaluates Eq.51 using P_lin, P_nw, P_w=P_lin-P_nw, and supplied ir_sigma2; Btree is the Eq.41 tree limit.\"\n";
    } else if (opts.backend == "native_cpp") {
        print_json_string("MARISA-B native C++ kernels/integrands with ReACT Quadrature and PowerSpectrum infrastructure");
        std::cout << ",\n";
        std::cout << "    \"note\": \"Native backend does not call BSPT::Bloop/Bloopterms or SPT::P13_dd.\"\n";
    } else {
        print_json_string("ReACT/Copter BSPT GR path patched with component diagnostics");
        std::cout << ",\n";
        std::cout << "    \"note\": \"Bootstrap backend for MARISA-B schema and Python/C++ boundary.\"\n";
    }
    std::cout << "  },\n";
    std::cout << "  \"results\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        print_component_json(rows[i], static_cast<int>(i), halo_bias_v1);
        if (i + 1 != rows.size()) std::cout << ",";
        std::cout << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);
        const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

        array k_arr;
        array p_arr;
        read_pk_table(opts.pk_table, k_arr, p_arr);
        array png_k_arr;
        array png_p_arr;
        array p_phi_arr;
        array m_arr;
        array nw_k_arr;
        array nw_p_arr;
        const bool use_png = opts.mode == "pre_recon_local_png_tree" || opts.mode == "pre_recon_local_png_1loop" || opts.mode == "post_recon_local_png_1loop" || opts.mode == "pre_recon_local_png_b112ii_diagnostic" || opts.mode == "post_recon_local_png_b112ii_diagnostic" || opts.mode == "pre_recon_local_png_finite_1loop" || opts.mode == "post_recon_local_png_finite_1loop" || opts.mode == "pre_recon_halo_bias_v1_local_png_tree" || opts.mode == "post_recon_halo_bias_v1_local_png_tree" || opts.mode == "pre_recon_halo_bias_v1_local_png_1loop_truncated" || opts.mode == "post_recon_halo_bias_v1_local_png_1loop_truncated";
        const bool use_nowiggle = opts.mode == "post_recon_gaussian_ir_nowiggle";
        if (use_png) {
            read_png_transfer_table(opts.png_table, png_k_arr, png_p_arr, p_phi_arr, m_arr);
        }
        if (use_nowiggle) {
            read_pk_table(opts.nowiggle_table, nw_k_arr, nw_p_arr);
        }

        Cosmology cosmology(0.6711, 0.9624, 0.3175, 0.049);
        cosmology.sigma8 = 0.834;
        InterpolatedPS plin(cosmology, k_arr, p_arr, true);
        InterpolatedPS png_plin(cosmology, use_png ? png_k_arr : k_arr, use_png ? png_p_arr : p_arr, true);
        InterpolatedPS transfer_m(cosmology, use_png ? png_k_arr : k_arr, use_png ? m_arr : p_arr, true);
        InterpolatedPS pnw(cosmology, use_nowiggle ? nw_k_arr : k_arr, use_nowiggle ? nw_p_arr : p_arr, true);
        BSPT bspt(cosmology, plin, opts.epsrel);
        set_unity_growth();

        ReactGrBootstrapBackend react_backend(bspt);
        NativeCppBackend native_backend(plin, opts);
        NativePostReconBackend post_backend(plin, opts);
        NativeHaloBiasV1GaussianBackend halo_bias_v1_pre_backend(plin, opts, false);
        NativeHaloBiasV1GaussianBackend halo_bias_v1_post_backend(plin, opts, true);
        NativeHaloBiasV1PngTreeBackend halo_bias_v1_png_tree_pre_backend(png_plin, transfer_m, opts, false);
        NativeHaloBiasV1PngTreeBackend halo_bias_v1_png_tree_post_backend(png_plin, transfer_m, opts, true);
        NativeHaloBiasV1PngOneLoopTruncatedBackend halo_bias_v1_png_1loop_pre_backend(
            png_plin, transfer_m, opts, false);
        NativeHaloBiasV1PngOneLoopTruncatedBackend halo_bias_v1_png_1loop_post_backend(
            png_plin, transfer_m, opts, true);
        NativePostReconIRNoWiggleBackend post_ir_backend(plin, pnw, opts);
        NativePngTreeBackend png_backend(png_plin, transfer_m, opts.b1);
        NativePngOneLoopBackend png_1loop_backend(png_plin, transfer_m, opts);
        NativePostReconPngOneLoopBackend post_png_1loop_backend(png_plin, transfer_m, opts);
        NativePngB112IIDiagnosticBackend png_b112ii_pre_backend(
            png_plin, transfer_m, opts, false);
        NativePngB112IIDiagnosticBackend png_b112ii_post_backend(
            png_plin, transfer_m, opts, true);
        NativePngFiniteOneLoopBackend png_finite_pre_backend(
            png_plin, transfer_m, opts, false);
        NativePngFiniteOneLoopBackend png_finite_post_backend(
            png_plin, transfer_m, opts, true);
        const BispectrumBackend* selected_backend = &react_backend;
        if (opts.mode == "pre_recon_halo_bias_v1_gaussian") {
            selected_backend = &halo_bias_v1_pre_backend;
        } else if (opts.mode == "post_recon_halo_bias_v1_gaussian") {
            selected_backend = &halo_bias_v1_post_backend;
        } else if (opts.mode == "pre_recon_halo_bias_v1_local_png_tree") {
            selected_backend = &halo_bias_v1_png_tree_pre_backend;
        } else if (opts.mode == "post_recon_halo_bias_v1_local_png_tree") {
            selected_backend = &halo_bias_v1_png_tree_post_backend;
        } else if (opts.mode == "pre_recon_halo_bias_v1_local_png_1loop_truncated") {
            selected_backend = &halo_bias_v1_png_1loop_pre_backend;
        } else if (opts.mode == "post_recon_halo_bias_v1_local_png_1loop_truncated") {
            selected_backend = &halo_bias_v1_png_1loop_post_backend;
        } else if (opts.mode == "pre_recon_local_png_tree") {
            selected_backend = &png_backend;
        } else if (opts.mode == "pre_recon_local_png_1loop") {
            selected_backend = &png_1loop_backend;
        } else if (opts.mode == "post_recon_local_png_1loop") {
            selected_backend = &post_png_1loop_backend;
        } else if (opts.mode == "pre_recon_local_png_b112ii_diagnostic") {
            selected_backend = &png_b112ii_pre_backend;
        } else if (opts.mode == "post_recon_local_png_b112ii_diagnostic") {
            selected_backend = &png_b112ii_post_backend;
        } else if (opts.mode == "pre_recon_local_png_finite_1loop") {
            selected_backend = &png_finite_pre_backend;
        } else if (opts.mode == "post_recon_local_png_finite_1loop") {
            selected_backend = &png_finite_post_backend;
        } else if (opts.mode == "post_recon_gaussian_ir_nowiggle") {
            selected_backend = &post_ir_backend;
        } else if (opts.mode == "post_recon_gaussian") {
            selected_backend = &post_backend;
        } else if (opts.backend == "native_cpp") {
            selected_backend = &native_backend;
        }
        MarisaBEngine engine(*selected_backend);
        const std::vector<ComponentResult> rows = engine.compute_many(opts.triangles);

        const std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();
        print_payload_json(opts, rows, wall_seconds);
    } catch (const std::exception& exc) {
        std::cerr << "marisa_b_triangle error: " << exc.what() << '\n';
        return 2;
    }
    return 0;
}

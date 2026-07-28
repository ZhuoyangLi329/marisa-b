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

namespace {

struct Triangle {
    double k1;
    double k2;
    double mu12;
};

struct Options {
    std::string pk_table;
    std::vector<Triangle> triangles;
    double epsrel = 1.0e-2;
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
        } else {
            std::stringstream msg;
            msg << "Unknown or incomplete argument: " << arg;
            throw std::runtime_error(msg.str());
        }
    }
    if (opts.pk_table.empty()) throw std::runtime_error("--pk-table is required");
    if (opts.triangles.empty()) opts.triangles.push_back(parse_triangle("0.05,0.05,-0.5"));
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

void set_unity_growth() {
    // 审计约定：输入的 P_L(k) 已经在目标红移 z=0.5。
    // 因此 ReACT/Copter 内部的 growth normalization 全部固定为 1，
    // 避免把目标红移功率谱再乘一次 D(z)/D(0)。
    Dl_spt = 1.0;
    D_spt = 1.0;
    dnorm_spt = 1.0;
    fl_spt = 1.0;
    fdgp_spt = 1.0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options opts = parse_args(argc, argv);

        array k_arr;
        array p_arr;
        read_pk_table(opts.pk_table, k_arr, p_arr);

        Cosmology cosmology(0.6711, 0.9624, 0.3175, 0.049);
        cosmology.sigma8 = 0.834;
        InterpolatedPS plin(cosmology, k_arr, p_arr, true);
        BSPT bspt(cosmology, plin, opts.epsrel);
        set_unity_growth();

        std::cout << std::setprecision(17);
        std::cout << "k1 k2 mu12 k3 Btree Btotal_react Bloopterms_react B321II_react B1loop_react "
                  << "B222_react B321I_react B411_react epsrel growth_mode\n";
        for (std::vector<Triangle>::const_iterator it = opts.triangles.begin(); it != opts.triangles.end(); ++it) {
            const double k3 = std::sqrt(it->k1 * it->k1 + it->k2 * it->k2 + 2.0 * it->k1 * it->k2 * it->mu12);
            const double btree = bspt.Btree(1, it->k1, it->k2, it->mu12);
            const double btotal = bspt.Bloop(1, it->k1, it->k2, it->mu12);
            const double bloopterms = bspt.Bloopterms(1, it->k1, it->k2, k3, it->mu12);
            double b222 = 0.0;
            double b321i = 0.0;
            double b411 = 0.0;
            bspt.BlooptermComponentsGR(it->k1, it->k2, k3, it->mu12, b222, b321i, b411);
            const double b1loop = btotal - btree;
            const double b321ii = btotal - btree - bloopterms;
            std::cout << it->k1 << ' ' << it->k2 << ' ' << it->mu12 << ' ' << k3 << ' '
                      << btree << ' ' << btotal << ' ' << bloopterms << ' ' << b321ii << ' '
                      << b1loop << ' ' << b222 << ' ' << b321i << ' ' << b411 << ' '
                      << opts.epsrel << " unity\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "react_bspt_driver error: " << exc.what() << '\n';
        return 2;
    }
    return 0;
}

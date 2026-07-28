#ifndef MARISA_B_HALO_V1_H
#define MARISA_B_HALO_V1_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class PowerSpectrum;

namespace marisa_b_halo_v1 {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct BiasPoint {
    double b1 = 1.0;
    double b2 = 0.0;
    double bK2 = 0.0;
};

struct Monomial {
    std::uint8_t b1;
    std::uint8_t b2;
    std::uint8_t bK2;
};

struct PolynomialTerm {
    Monomial powers;
    double coefficient;
};

class Polynomial {
public:
    Polynomial() noexcept = default;
    Polynomial(const Polynomial& other) noexcept;
    Polynomial(Polynomial&& other) noexcept;
    Polynomial& operator=(const Polynomial& other) noexcept;
    Polynomial& operator=(Polynomial&& other) noexcept;

    static Polynomial constant(double value);
    static Polynomial variable_b1();
    static Polynomial variable_b2();
    static Polynomial variable_bK2();

    double evaluate(const BiasPoint& bias) const;
    double coefficient(int b1_power, int b2_power, int bK2_power) const;
    bool empty() const;
    std::vector<PolynomialTerm> terms() const;
    std::string expression() const;

    Polynomial& operator+=(const Polynomial& other);
    Polynomial& operator-=(const Polynomial& other);
    Polynomial& operator*=(double scalar);

private:
    static constexpr std::size_t kMaximumTerms = 84;
    static constexpr int kMaximumTotalDegree = 6;

    void append(Monomial powers, double coefficient);

    std::array<PolynomialTerm, kMaximumTerms> terms_;
    std::uint8_t size_ = 0;

    friend Polynomial operator+(Polynomial left, const Polynomial& right);
    friend Polynomial operator-(Polynomial left, const Polynomial& right);
    friend Polynomial operator*(const Polynomial& left, const Polynomial& right);
    friend Polynomial operator*(Polynomial polynomial, double scalar);
    friend Polynomial operator*(double scalar, Polynomial polynomial);
};

Polynomial operator+(Polynomial left, const Polynomial& right);
Polynomial operator-(Polynomial left, const Polynomial& right);
Polynomial operator*(const Polynomial& left, const Polynomial& right);
Polynomial operator*(Polynomial polynomial, double scalar);
Polynomial operator*(double scalar, Polynomial polynomial);

struct TadpoleTerm {
    double external_k = 0.0;
    Polynomial coefficient;
};

struct KernelTemplate {
    Polynomial regular;
    std::vector<TadpoleTerm> matter_f3_tadpoles;
};

struct ReconstructionConfig {
    bool enabled = false;
    double smoothing_radius = 15.0;
    double bias_recon = 2.7340475186190334;
    double cell_size = 8.0;
};

enum class RadialCoordinate {
    Logarithmic,
    Linear,
};

struct IntegrationConfig {
    double qmin = 1.0e-4;
    double qmax = 0.0;
    int n_radial = 24;  // nodes in each interval split at the three external k values
    int n_mu = 16;
    int n_phi = 16;
    double p13_epsrel = 1.0e-6;
    double p13_epsabs = 1.0e-12;
    RadialCoordinate radial_coordinate = RadialCoordinate::Logarithmic;
    ReconstructionConfig reconstruction;
};

struct Triangle {
    double k1 = 0.05;
    double k2 = 0.06;
    double k3 = 0.07;
};

struct CanonicalTriangle {
    Triangle sides;
    Vec3 k1;
    Vec3 k2;
    Vec3 k3;
};

struct ComponentTemplates {
    CanonicalTriangle triangle;
    Polynomial tree;
    Polynomial B222;
    Polynomial B321I;
    Polynomial B321II;
    Polynomial B411;
    Polynomial one_loop;
    Polynomial total;
    Polynomial stochastic_alpha3_raw;
    double stochastic_alpha4_raw = 1.0;
    std::size_t loop_nodes = 0;
};

struct ComponentValues {
    CanonicalTriangle triangle;
    double tree = 0.0;
    double B222 = 0.0;
    double B321I = 0.0;
    double B321II = 0.0;
    double B411 = 0.0;
    double one_loop = 0.0;
    double total = 0.0;
    double stochastic_alpha3_raw = 0.0;
    double stochastic_alpha4_raw = 1.0;
    std::size_t loop_nodes = 0;
};

Vec3 add(const Vec3& left, const Vec3& right);
Vec3 subtract(const Vec3& left, const Vec3& right);
Vec3 negate(const Vec3& vector);
double dot(const Vec3& left, const Vec3& right);
double norm(const Vec3& vector);
double tidal_s2(const Vec3& left, const Vec3& right);

CanonicalTriangle canonicalize_triangle(const Triangle& triangle);
CanonicalTriangle canonicalize_closed_vectors(const std::array<Vec3, 3>& vectors);
double reconstruction_window(const Vec3& momentum, const ReconstructionConfig& config);
double reconstruction_shift_factor(
    const Vec3& output_momentum,
    const Vec3& block_momentum,
    const ReconstructionConfig& config);

KernelTemplate pre_reconstruction_kernel(const std::vector<Vec3>& momenta);
KernelTemplate reconstructed_kernel(
    const std::vector<Vec3>& momenta,
    const ReconstructionConfig& config);

ComponentTemplates compute_templates(
    const PowerSpectrum& linear_power,
    const Triangle& triangle,
    const IntegrationConfig& config);

ComponentTemplates compute_templates_vectors(
    const PowerSpectrum& linear_power,
    const std::array<Vec3, 3>& closed_vectors,
    const IntegrationConfig& config);

ComponentValues compute_direct(
    const PowerSpectrum& linear_power,
    const Triangle& triangle,
    const IntegrationConfig& config,
    const BiasPoint& bias);

ComponentValues compute_direct_vectors(
    const PowerSpectrum& linear_power,
    const std::array<Vec3, 3>& closed_vectors,
    const IntegrationConfig& config,
    const BiasPoint& bias);

ComponentValues evaluate_templates(
    const ComponentTemplates& templates,
    const BiasPoint& bias);

double add_residual_stochastic(
    double deterministic_total,
    double alpha3_raw_basis,
    double alpha4_raw_basis,
    double alpha3,
    double alpha4,
    double number_density);

}  // namespace marisa_b_halo_v1

#endif  // MARISA_B_HALO_V1_H

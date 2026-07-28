# Gaussian halo bias-v1 derivation

## Field and kernel convention

The real-space halo field is the fixed truncated functional

\[
\delta_h=b_1\delta+\frac{b_2}{2}\delta^2+b_{K^2}K_{ij}K_{ij},
\qquad
K_{ij}(\boldsymbol k)=
\left(\frac{k_i k_j}{k^2}-\frac{\delta_{ij}}{3}\right)\delta(\boldsymbol k).
\]

No other deterministic operator is present.  With symmetric EdS matter kernels
\(F_n\),

\[
\delta_h^{(n)}(\boldsymbol k)=
\int_{\boldsymbol k_{1\cdots n}=\boldsymbol k}
K_n(\boldsymbol k_1,\ldots,\boldsymbol k_n)
\prod_{a=1}^n\delta_{\rm L}(\boldsymbol k_a),
\]

and

\[
K_n=b_1F_n+b_2D_n+b_{K^2}T_n.
\]

For a subset \(A\subset[n]\), write \(\boldsymbol k_A=\sum_{a\in A}\boldsymbol
k_a\), \(A^c=[n]\setminus A\), and

\[
S_2(\boldsymbol p,\boldsymbol r)
=\left(\widehat{\boldsymbol p}\!\cdot\!\widehat{\boldsymbol r}\right)^2-\frac13.
\]

Direct expansion of the two quadratic operators gives

\[
D_n=\frac12\sum_{m=1}^{n-1}\binom{n}{m}^{-1}
\sum_{\substack{A\subset[n]\\|A|=m}}
F_m(A)F_{n-m}(A^c),
\]

\[
T_n=\sum_{m=1}^{n-1}\binom{n}{m}^{-1}
\sum_{\substack{A\subset[n]\\|A|=m}}
S_2(\boldsymbol k_A,\boldsymbol k_{A^c})F_m(A)F_{n-m}(A^c).
\]

Thus \(D_1=T_1=0\), \(D_2=1/2\), and
\(T_2=S_2(\boldsymbol k_1,\boldsymbol k_2)\).  The subset form is used in
the implementation through fourth order, so no separately hand-coded third-
or fourth-order bias formula can acquire a different symmetrization.

The existing MARISA-B dark-matter implementation supplies \(F_2,F_3,F_4\)
and the matter \(F_3(\boldsymbol k,\boldsymbol q,-\boldsymbol q)\) tadpole
integral.  Its source is not changed by the halo implementation.

## Reconstruction map

Standard reconstruction shifts the tracer and random fields by the same
displacement.  Their difference is

\[
\delta_{h,{\rm rec}}(\boldsymbol k)=
\int d^3x\,e^{-i\boldsymbol k\cdot\boldsymbol x}
e^{-i\boldsymbol k\cdot\boldsymbol s(\boldsymbol x)}\delta_h(\boldsymbol x),
\]

where each perturbative displacement block of momentum \(\boldsymbol p\)
contributes

\[
R_{\boldsymbol k}(\boldsymbol p)K_{|B|}(B),\qquad
R_{\boldsymbol k}(\boldsymbol p)
=-\frac{\boldsymbol k\cdot\boldsymbol p}{p^2}
\frac{W_{\rm rec}(\boldsymbol p)}{b_{\rm rec}}.
\]

For a set partition \(\pi\) of the \(n\) labelled linear legs, one block is
the density block and all other blocks are displacement blocks.  The fully
symmetric reconstructed kernel is

\[
K_n^{\rm rec}(1,\ldots,n)=
\sum_{\pi\in\Pi_n}\frac{\prod_{C\in\pi}|C|!}{n!}
\sum_{A\in\pi}K_{|A|}(A)
\prod_{\substack{B\in\pi\\B\ne A}}
R_{\boldsymbol k}(\boldsymbol k_B)K_{|B|}(B),
\qquad \boldsymbol k=\boldsymbol k_{[n]}.
\]

The factor \(\prod_C|C|!/n!\) is the complete symmetrization factor.  For
\(n=2,3,4\), this reproduces respectively the coefficients
\(1/2\), \((1/3,1/6)\), and \((1/4,1/6,1/12,1/24)\) in the explicit
post-reconstruction kernels.  The Fourier zero mode of the measured overdensity
does not generate a displacement or a density block.

For the standard-JAXRecon setup in the data contract,

\[
W_{\rm rec}(\boldsymbol p)=e^{-p^2R^2/2}
\prod_{a=x,y,z}\operatorname{sinc}^4\!\left(\frac{p_a\Delta}{2}\right),
\]

with \(R=15\,h^{-1}{\rm Mpc}\), \(\Delta=8\,h^{-1}{\rm Mpc}\), and
\(b_{\rm rec}=2.7340475186190334\).  The fourth power per Cartesian axis is
the combined CIC density-paint and CIC displacement-read transfer.  Setting
the reconstruction map off, or taking \(R\to\infty\), returns \(K_n\).

## Gaussian bispectrum through one loop

Let \(\boldsymbol k_1+\boldsymbol k_2+\boldsymbol k_3=0\),
\(P_i=P_{\rm L}(k_i)\), and let every \(K_n\) below mean either consistently
pre- or post-reconstruction kernels.  The five retained terms are

\[
B_{\rm tree}=2\sum_{i<j}K_1(i)K_1(j)K_2(i,j)P_iP_j,
\]

\[
B_{222}=8\int_{\boldsymbol q}
K_2(\boldsymbol q,\boldsymbol k_1-\boldsymbol q)
K_2(-\boldsymbol q,\boldsymbol k_2+\boldsymbol q)
K_2(\boldsymbol q-\boldsymbol k_1,-\boldsymbol q-\boldsymbol k_2)
P(q)P(|\boldsymbol k_1-\boldsymbol q|)P(|\boldsymbol k_2+\boldsymbol q|),
\]

\[
B_{321}^{I}=6\sum_{i\ne j}K_1(i)P_i\int_{\boldsymbol q}
K_2(\boldsymbol q,\boldsymbol k_j-\boldsymbol q)
K_3(\boldsymbol k_i,\boldsymbol q,\boldsymbol k_j-\boldsymbol q)
P(q)P(|\boldsymbol k_j-\boldsymbol q|),
\]

\[
B_{321}^{II}=6\sum_{i<j}K_2(i,j)P_iP_j
\left[K_1(i)I_3(k_j)+K_1(j)I_3(k_i)\right],
\quad
I_3(k)=\int_{\boldsymbol q}K_3(\boldsymbol k,\boldsymbol q,-\boldsymbol q)P(q),
\]

\[
B_{411}=12\sum_{i<j}K_1(i)K_1(j)P_iP_j
\int_{\boldsymbol q}K_4(\boldsymbol k_i,\boldsymbol k_j,
\boldsymbol q,-\boldsymbol q)P(q),
\qquad
\int_{\boldsymbol q}\equiv\int\frac{d^3q}{(2\pi)^3}.
\]

The regulated functional first sorts the three external side lengths and uses
the two shortest sides as \(\boldsymbol k_1,\boldsymbol k_2\); the longest side
closes the triangle.  This canonical routing is applied before every loop.
The loop domain is the same sphere \(q_{\min}\le q\le q_{\max}\) for every
component.  With \(u=\ln q\), \(\mu=\widehat{\boldsymbol q}\cdot
\widehat{\boldsymbol k}_1\), and azimuth \(\varphi\),

\[
\int_{\boldsymbol q}f(\boldsymbol q)=
\frac{1}{(2\pi)^3}\int_{\ln q_{\min}}^{\ln q_{\max}}du
\int_{-1}^{1}d\mu\int_0^{2\pi}d\varphi\;q^3 f(\boldsymbol q).
\]

The factor \(q^3\) is retained explicitly.  All coefficients of every
monomial in \((b_1,b_2,b_{K^2})\) are accumulated on the same deterministic
cubature nodes.  The logarithmic radial interval is split at every external
side \(k_i\) that lies inside the cutoff, and the configured Gauss--Legendre
order is applied to each subinterval.  This resolves the integrable soft
regions near \(q=k_i\) without changing the common spherical cutoff.  A
separately implemented linear-\(q\) rule with the \(q^2dq\) Jacobian is retained
only as a numerical cross-check of the production log-\(q\) rule.

The caller must choose \(q_{\max}\) explicitly.  In this truncated,
unrenormalized halo functional it is a regulator that changes the model, not a
cubature tolerance: a \(q_{\max}\) scan must therefore be reported rather than
misidentified as an integration-error estimate.

Exact opposite-pair matter tadpoles are not evaluated by inserting a singular
\(F_3(\boldsymbol k,\boldsymbol q,-\boldsymbol q)\) point into the kernel.
They are kept symbolically and integrated with the existing MARISA-B matter
\(P_{13}\) result,

\[
\int_{\boldsymbol q}F_3(\boldsymbol k,\boldsymbol q,-\boldsymbol q)P(q)
=\frac{P_{13}(k)}{6P_{\rm L}(k)}.
\]

## Residual stochastic basis

The estimator has already subtracted its Poisson bispectrum.  Only the two
residual non-Poisson amplitudes are added at the fitting layer:

\[
B_{\rm stoch}=\frac{\alpha_3}{\bar n}
b_1^2\left[P_1+P_2+P_3\right]
+\frac{\alpha_4}{\bar n^2}.
\]

The core therefore returns the raw bases
\(S_3=b_1^2(P_1+P_2+P_3)\) and \(S_4=1\), without multiplying by
\(\alpha_3,\alpha_4\), or \(\bar n\).

## Primary references

- D. Jeong, *Cosmology with high-redshift galaxy survey* (dissertation),
  Chapter 2, especially Eqs. (2.22)--(2.31).
- M. Shirasaki et al., arXiv:2010.04567v2, Eqs. (1), (8)--(12).
- N. Sugiyama, arXiv:2403.18262v1, Eq. (45) and Appendix A.
- N. Sugiyama, arXiv:2508.17331v2, the Eulerian reconstruction mapping and
  SPT/bias discussion.
- A. Barreira, arXiv:2107.06887v3, Eqs. (1.3) and (A.4)--(A.5).

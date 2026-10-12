#ifndef MARISA_B_FINITE_PNG_POLY2_H
#define MARISA_B_FINITE_PNG_POLY2_H
// 模块大纲：解析 fNL 二阶多项式；Q 是系数而不是二阶导数。
#include <stdexcept>
namespace marisa_b_native {
struct FnlPoly2 { double c0=0.,c1=0.,c2=0.; };
inline FnlPoly2 operator+(FnlPoly2 a,FnlPoly2 b) {return {a.c0+b.c0,a.c1+b.c1,a.c2+b.c2};}
inline FnlPoly2 operator-(FnlPoly2 a,FnlPoly2 b) {return {a.c0-b.c0,a.c1-b.c1,a.c2-b.c2};}
inline FnlPoly2 operator*(FnlPoly2 a,FnlPoly2 b) {
    return {a.c0*b.c0,a.c0*b.c1+a.c1*b.c0,a.c0*b.c2+a.c1*b.c1+a.c2*b.c0};
}
inline FnlPoly2 operator*(double a,FnlPoly2 b) {return {a*b.c0,a*b.c1,a*b.c2};}
inline FnlPoly2 reciprocal(FnlPoly2 a) {
    if(a.c0==0.) throw std::invalid_argument("zero reconstruction denominator");
    return {1./a.c0,-a.c1/(a.c0*a.c0),a.c1*a.c1/(a.c0*a.c0*a.c0)-a.c2/(a.c0*a.c0)};
}
inline FnlPoly2 operator/(FnlPoly2 a,FnlPoly2 b) {return a*reciprocal(b);}
}
#endif

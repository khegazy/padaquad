"""
Gauss-Kronrod embedded quadrature pairs.

A pair (G_n, K_{2n+1}) is an n-point Gauss-Legendre rule plus an
(n+1)-point Kronrod extension. K_{2n+1} integrates polynomials exactly up
to degree 3n+1 for even n and 3n+2 for odd n (gk7, gk15, gk21, gk31:
11, 23, 31, 47); the pair's error indicator is K_{2n+1} - G_n. The degrees
are measured from the stored tables by tests/test_exactness.py.

Where the digits come from:

- gk15, gk21, gk31: the QUADPACK routines dqk15.f, dqk21.f and dqk31.f on
  Netlib (https://www.netlib.org/quadpack/, revision 830518), whose comments
  say the values were "evaluated with 80 decimal digit arithmetic by
  L. W. Fullerton, Bell Labs, Nov. 1981" (Piessens, de Doncker-Kapenga,
  Ueberhuber and Kahaner, QUADPACK, Springer 1983). Every digit of the 33-digit
  values below matches those files (compared by script on 2026-11-12).
- gk7: QUADPACK has no 7-point routine. The 30-digit values below agree with
  an 80-digit computation of the Kronrod extension of the 3-point Gauss rule
  (the roots of the Stieltjes polynomial) to all printed digits. Patterson
  1968 (Math. Comp. 22(104):847-856) tabulates this rule; its table was not
  available to check.

Python keeps float64 precision when it reads these numbers. The degree of every
table is checked in tests/test_exactness.py, and the Gauss parts agree with
numpy's Gauss-Legendre rule in tests/test_tableau.py.
"""

from __future__ import annotations

import torch

from ._base import UniformMethodClass, _Tableau


def _build_gauss_kronrod_tableau(
    xgk_half_with_zero: list[float],
    wgk_half_with_zero: list[float],
    wg_at_g_pos: list[float],
    g_pos_indices_in_half: list[int],
    wg_at_center: float = 0.0,
) -> _Tableau:
    """Construct a Gauss-Kronrod _Tableau on [0, 1].

    Inputs are tabulated for the canonical interval (-1, 1) with values
    given for the positive half plus the center node (which is the last
    entry of ``xgk_half_with_zero``). The full K_{2n+1} arrays are built
    by mirroring the positive-half about 0.

    Args:
        xgk_half_with_zero: K_{2n+1} node positions in (-1, 1), positive
            half (descending magnitude) followed by the center 0.0.
            Length n + 1. For K21 this has 11 entries: 10 positive +
            center.
        wgk_half_with_zero: K_{2n+1} weights at the corresponding nodes.
            Length n + 1.
        wg_at_g_pos: G_n weights at the G_n positive-half nodes.
            Length n // 2 (Kronrod's theorem: G_n has n nodes symmetric
            about 0; for even n there are n // 2 distinct positive
            magnitudes; for odd n there are (n - 1) // 2 distinct
            positive magnitudes plus the center).
        g_pos_indices_in_half: indices into the positive half of
            ``xgk_half_with_zero`` where G_n nodes coincide with K
            extension nodes. Length matches ``wg_at_g_pos``.
        wg_at_center: G_n weight at the center node 0.0. Use 0.0 if
            G_n has no center node (even n: G10, G14, ...). Use the
            published center weight for odd n: G7, G15, ...

    Returns:
        _Tableau with c (K nodes mapped to [0, 1]), b (K weights, the
        high-order estimate), and b_error (K - G_extended, where
        G_extended has 0 at K-only nodes — the standard error
        indicator for an embedded G-K pair).
    """
    # Every tensor is created as float64 explicitly: a tensor built from Python
    # floats without dtype= is float32, which would silently round the weights.
    f64 = torch.float64
    xgk_half = torch.tensor(xgk_half_with_zero, dtype=f64)  # [n+1]
    wgk_half = torch.tensor(wgk_half_with_zero, dtype=f64)  # [n+1]
    wg_g = torch.tensor(wg_at_g_pos, dtype=f64)
    zero = torch.zeros(1, dtype=f64)
    one = torch.ones(1, dtype=f64)

    # Build full K_{2n+1} arrays in (-1, 1), ascending order:
    #   [-x_n, ..., -x_1, 0, x_1, ..., x_n]
    # The positive half is xgk_half[:-1] in DESCENDING magnitude order
    # (largest first); reversed, it is in ascending order.
    pos_x = xgk_half[:-1]  # descending magnitudes
    pos_w = wgk_half[:-1]
    center_w = wgk_half[-1:]

    # Negative side (ascending nodes from most negative): -pos_x in original
    # order is [-largest, -second_largest, ..., -smallest], which is exactly
    # ascending from most negative to least negative.
    xgk_full = torch.cat([-pos_x, zero, pos_x.flip(0)])
    wgk_full = torch.cat([pos_w, center_w, pos_w.flip(0)])

    # Build G_n weights extended to the K_{2n+1} grid (zero at K-only
    # nodes). For even n, G_n has no center node (wg_at_center = 0).
    # For odd n (G7, G15, ...), the center node IS a G_n node and
    # wg_at_center is its non-zero weight.
    wg_extended_pos_half = torch.zeros_like(pos_w)
    wg_extended_pos_half[g_pos_indices_in_half] = wg_g
    wg_extended_full = torch.cat(
        [
            wg_extended_pos_half,
            torch.tensor([wg_at_center], dtype=f64),
            wg_extended_pos_half.flip(0),
        ]
    )

    # Map (-1, 1) -> [0, 1]: c = (x + 1) / 2, w = w / 2 (Jacobian).
    c_unit = (xgk_full + 1.0) / 2.0
    b_unit = wgk_full / 2.0
    b_g_unit = wg_extended_full / 2.0

    # The parallel solver's _RK_integral computes the per-panel step size
    # as ``h = t[:, -1] - t[:, 0]`` and assumes ``c[0] == 0`` and
    # ``c[-1] == 1``. Gauss-Kronrod nodes are interior (no endpoints) so
    # without padding ``h`` would equal 0.99566... instead of 1.0 and the
    # integral would be off by that factor. Standard fix: pad with two
    # zero-weight nodes at 0 and 1; they're evaluated but contribute
    # nothing to the integral or error estimate. The cost is 2 wasted
    # f evaluations per panel — negligible for an N+2 = 23-node rule.
    c_padded = torch.cat([zero, c_unit, one])
    b_padded = torch.cat([zero, b_unit, zero])
    b_g_padded = torch.cat([zero, b_g_unit, zero])
    return _Tableau(c=c_padded, b=b_padded, b_error=b_padded - b_g_padded)


# G10-K21 (Gauss-Kronrod 21-point). K21 degree = 3n+1 = 31 (n = 10, even).
# Source: Netlib QUADPACK dqk21.f (xgk, wgk, wg). Positive half-axis nodes
# followed by the center node 0.0.
_GK21_XGK_HALF = [
    0.995657163025808080735527280689003,
    0.973906528517171720077964012084452,
    0.930157491355708226001207180059508,
    0.865063366688984510732096688423493,
    0.780817726586416897063717578345042,
    0.679409568299024406234327365114874,
    0.562757134668604683339000099272694,
    0.433395394129247190799265943165784,
    0.294392862701460198131126603103866,
    0.148874338981631210884826001129720,
    0.0,
]
_GK21_WGK_HALF = [
    0.011694638867371874278064396062192,
    0.032558162307964727478818972459390,
    0.054755896574351996031381300244580,
    0.075039674810919952767043140916190,
    0.093125454583697605535065465083366,
    0.109387158802297641899210590325805,
    0.123491976262065851077958109831074,
    0.134709217311473325928054001771707,
    0.142775938577060080797094273138717,
    0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
]
# G10 weights at G10 positive nodes. G10 nodes coincide with K21 nodes
# at indices 1, 3, 5, 7, 9 of the positive half (the "G" rows in the
# Patterson-style Kronrod construction).
_GK21_WG = [
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
]
_GK21_G_INDICES = [1, 3, 5, 7, 9]

_GAUSS_KRONROD_21 = UniformMethodClass(
    degree=31,  # K21: 3n+1 with n = 10 (even); order (convergence rate) 32
    error_exponent=21,  # estimate dominated by G10, exactness 2n-1 = 19
    tableau=_build_gauss_kronrod_tableau(
        _GK21_XGK_HALF,
        _GK21_WGK_HALF,
        _GK21_WG,
        _GK21_G_INDICES,
        wg_at_center=0.0,  # G10 (even n) has no center node
    ),
)


# G3-K7 (Gauss-Kronrod 7-point). K7 degree = 3n+2 = 11 (n = 3, odd).
# G3 (odd n=3) has a center node — note wg_at_center below.
# Computed to 80 digits (see the module docstring); tabulated in Patterson 1968.
_GK7_XGK_HALF = [
    0.960491268708020283423507092629,
    0.774596669241483377035853079956,  # G3 node = sqrt(3/5)
    0.434243749346802558002071502845,
    0.0,  # center, also a G3 node
]
_GK7_WGK_HALF = [
    0.104656226026467265193823857192,
    0.268488089868333440728569280666,
    0.401397414775962222905051818618,
    0.450916538658474142345110087045,
]
# G3 weight at the positive G3 node (K7 positive-half index 1). G3 weight = 5/9.
_GK7_WG = [0.555555555555555555555555555556]
_GK7_G_INDICES = [1]
# G3 weight at the center node 0.0 = 8/9 (G3 has odd n, so it includes 0.0).
_GK7_WG_CENTER = 0.888888888888888888888888888889

_GAUSS_KRONROD_7 = UniformMethodClass(
    degree=11,  # K7: 3n+2 with n = 3 (odd); order (convergence rate) 12
    error_exponent=7,  # estimate dominated by G3, exactness 2n-1 = 5
    tableau=_build_gauss_kronrod_tableau(
        _GK7_XGK_HALF,
        _GK7_WGK_HALF,
        _GK7_WG,
        _GK7_G_INDICES,
        wg_at_center=_GK7_WG_CENTER,
    ),
)


# G7-K15 (Gauss-Kronrod 15-point). K15 degree = 3n+2 = 23 (n = 7, odd).
# G7 (odd n=7) has a center node — note wg_at_center below.
# Source: Netlib QUADPACK dqk15.f (xgk, wgk, wg).
_GK15_XGK_HALF = [
    0.991455371120812639206854697526329,
    0.949107912342758524526189684047851,  # G7 node
    0.864864423359769072789712788640926,
    0.741531185599394439863864773280788,  # G7 node
    0.586087235467691130294144838258730,
    0.405845151377397166906606412076961,  # G7 node
    0.207784955007898467600689403773245,
    0.0,  # center, also a G7 node
]
_GK15_WGK_HALF = [
    0.022935322010529224963732008058970,
    0.063092092629978553290700663189204,
    0.104790010322250183839876322541518,
    0.140653259715525918745189590510238,
    0.169004726639267902826583426598550,
    0.190350578064785409913256402421014,
    0.204432940075298892414161999234649,
    0.209482141084727828012999174891714,
]
# G7 weights at the three positive G7 nodes (K15 indices 1, 3, 5).
_GK15_WG = [
    0.129484966168869693270611432679082,
    0.279705391489276667901467771423780,
    0.381830050505118944950369775488975,
]
_GK15_G_INDICES = [1, 3, 5]
# G7 weight at the center node (G7 has odd n, so it includes 0.0).
_GK15_WG_CENTER = 0.417959183673469387755102040816327

_GAUSS_KRONROD_15 = UniformMethodClass(
    degree=23,  # K15: 3n+2 with n = 7 (odd); order (convergence rate) 24
    error_exponent=15,  # estimate dominated by G7, exactness 2n-1 = 13
    tableau=_build_gauss_kronrod_tableau(
        _GK15_XGK_HALF,
        _GK15_WGK_HALF,
        _GK15_WG,
        _GK15_G_INDICES,
        wg_at_center=_GK15_WG_CENTER,
    ),
)


# G15-K31 (Gauss-Kronrod 31-point). K31 degree = 3n+2 = 47 (n = 15, odd).
# G15 (odd n=15) has a center node.
# Source: Netlib QUADPACK dqk31.f (xgk, wgk, wg).
_GK31_XGK_HALF = [
    0.998002298693397060285172840152271,
    0.987992518020485428489565718586613,  # G15 node
    0.967739075679139134257347978784337,
    0.937273392400705904307758947710209,  # G15 node
    0.897264532344081900882509656454496,
    0.848206583410427216200648320774217,  # G15 node
    0.790418501442465932967649294817947,
    0.724417731360170047416186054613938,  # G15 node
    0.650996741297416970533735895313275,
    0.570972172608538847537226737253911,  # G15 node
    0.485081863640239680693655740232351,
    0.394151347077563369897207370981045,  # G15 node
    0.299180007153168812166780024266389,
    0.201194093997434522300628303394596,  # G15 node
    0.101142066918717499027074231447392,
    0.0,  # center, also a G15 node
]
_GK31_WGK_HALF = [
    0.005377479872923348987792051430128,
    0.015007947329316122538374763075807,
    0.025460847326715320186874001019653,
    0.035346360791375846222037948478360,
    0.044589751324764876608227299373280,
    0.053481524690928087265343147239430,
    0.062009567800670640285139230960803,
    0.069854121318728258709520077099147,
    0.076849680757720378894432777482659,
    0.083080502823133021038289247286104,
    0.088564443056211770647275443693774,
    0.093126598170825321225486872747346,
    0.096642726983623678505179907627589,
    0.099173598721791959332393173484603,
    0.100769845523875595044946662617570,
    0.101330007014791549017374792767493,
]
# G15 weights at positive G15 nodes (K31 half-indices 1, 3, 5, 7, 9, 11, 13).
_GK31_WG = [
    0.030753241996117268354628393577204,
    0.070366047488108124709267416450667,
    0.107159220467171935011869546685869,
    0.139570677926154314447804794511028,
    0.166269205816993933553200860481209,
    0.186161000015562211026800561866423,
    0.198431485327111576456118326443839,
]
_GK31_G_INDICES = [1, 3, 5, 7, 9, 11, 13]
_GK31_WG_CENTER = 0.202578241925561272880620199967519

_GAUSS_KRONROD_31 = UniformMethodClass(
    degree=47,  # K31: 3n+2 with n = 15 (odd); order (convergence rate) 48
    error_exponent=31,  # estimate dominated by G15, exactness 2n-1 = 29
    tableau=_build_gauss_kronrod_tableau(
        _GK31_XGK_HALF,
        _GK31_WGK_HALF,
        _GK31_WG,
        _GK31_G_INDICES,
        wg_at_center=_GK31_WG_CENTER,
    ),
)


GK_METHODS = {
    "gk7": _GAUSS_KRONROD_7,
    "gk15": _GAUSS_KRONROD_15,
    "gk21": _GAUSS_KRONROD_21,
    "gk31": _GAUSS_KRONROD_31,
}

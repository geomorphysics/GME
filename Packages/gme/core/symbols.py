"""
Mathematical Symbol used in GME equations.

Also their latex representation for pretty printing in notebooks.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`SymPy <sympy>`

---------------------------------------------------------------------
"""
from sympy.physics.units.systems import SI
from sympy.physics.units import length, time
from sympy import Symbol, MatrixSymbol, Function

# import logging
# logging.info('gme.core.Symbol')


i: Symbol = Symbol(r"i", real=True)
j: Symbol = Symbol(r"j", real=True)
k: Symbol = Symbol(r"k", real=True)
l: Symbol = Symbol(r"l", real=True)
m: Symbol = Symbol(r"m", real=True)
F: Symbol = Symbol(r"\mathcal{F}", real=True, positive=True)
Fstar: Symbol = Symbol(r"\mathcal{F}_*", real=True, positive=True)
Fstar_px_pz: Symbol = Symbol(
    r"\mathcal{F}_{*}(p_x\,p_z)", real=True, positive=True
)
H: Symbol = Symbol(r"\mathcal{H}", real=True, negative=False)
# ideally, both should be positive=True
L: Symbol = Symbol(r"\mathcal{L}", real=True, negative=False)
G1: Symbol = Symbol(r"G1")
G2: Symbol = Symbol(r"G2")
p: Symbol = Symbol(r"p", real=True, positive=True)
p_0: Symbol = Symbol(r"p_0", real=True, positive=True)
u: Symbol = Symbol(r"u", real=True)
px: Symbol = Symbol(r"p_x", real=True)
pz: Symbol = Symbol(r"p_z", real=True, negative=True)
pxhat: Symbol = Symbol(r"\hat{p}_x", real=True)
pzhat_0: Symbol = Symbol(r"\hat{p}_{z_0}", real=True, negative=True)
pzhat: Symbol = Symbol(r"\hat{p}_z", real=True, negative=True)
# pz_0: Symbol = Symbol(r"p_{z_0}", real=True)
px_min: Symbol = Symbol(r"p_{x_\text{min}}", real=True)
pz_min: Symbol = Symbol(r"p_{z_\text{min}}", real=True)
beta_max: Symbol = Symbol(r"\beta_{\text{max}}")
alpha_max: Symbol = Symbol(r"\alpha_{\text{max}}", real=True)
pxp: Symbol = Symbol(r"p_x^+", real=True, positive=True)
pzp: Symbol = Symbol(r"p_z^+", real=True, positive=True)
pzm: Symbol = Symbol(r"p_z^-", real=True, negative=True)
pz_0: Symbol = Symbol(r"p_{z_0}", real=True, negative=True)
px_0: Symbol = Symbol(r"p_{x_0}", real=True, positive=True)
c: Symbol = Symbol(r"c", real=True)

psqrd_substn: Symbol = px ** 2 + pz ** 2
# eta: Symbol = Symbol(r"\eta", real=True, positive=True)
mu: Symbol = Symbol(r"\mu", real=True, positive=True)
epsilon: Symbol = Symbol(r"\epsilon", real=True, positive=True)

t: Symbol = Symbol(r"t", real=True, negative=False)
# t_Lc: Symbol = Symbol(r't^{\rightarrow\mathrm{L_c}}',
# real=True, positive=True)
th_0p95: Symbol = Symbol(r"t^{\rightarrow{0.95}}", real=True, negative=False)
h_0p95: Symbol = Symbol(r"h_{0.95}", real=True, negative=False)
th_0p9: Symbol = Symbol(r"t^{\rightarrow{0.9}}", real=True, negative=False)
# h_0p9: Symbol = Symbol(r"h_{0.9}", real=True, negative=False)
t_oneyear: Symbol = Symbol(r"t_{\mathrm{1y}}", real=True, positive=True)
t_My: Symbol = Symbol(r"t_{\mathrm{My}}", real=True, positive=True)
that: Symbol = Symbol(r"\hat{t}", real=True, negative=False)
tv_0: Symbol = Symbol(r"t^{\downarrow_{0}}", real=True, positive=True)
th_0: Symbol = Symbol(r"t^{\rightarrow_{0}}", real=True, positive=True)

beta: Symbol = Symbol(r"\beta", real=True)
beta_: Symbol = Symbol(r"\beta_x", real=True)
beta_crit: Symbol = Symbol(r"\beta_c", real=True)
beta_0: Symbol = Symbol(r"\beta_0", real=True, positive=True)
SI.set_quantity_dimension(beta_0, 1)
betaplus: Symbol = Symbol(r"\beta^+", real=True, positive=True)
alpha: Symbol = Symbol(r"\alpha", real=True)
alpha_ext: Symbol = Symbol(r"\alpha_\mathrm{ext}", real=True)
alphaplus: Symbol = Symbol(r"\alpha^+", real=True, positive=True)
alpha_extremum: Symbol = Symbol(r"\alpha_\text{extremum}")
beta_at_alpha_extremum: Symbol = Symbol(r"\beta_{\text{extremum}\{\alpha\}}")
phi: Symbol = Symbol(r"\phi", real=True)
psi: Symbol = Symbol(r"\psi", real=True)

rvec: Symbol = Symbol(r"\mathbf{r}", real=True)
r: Symbol = Symbol(r"r", real=True, negative=False)
rx: Symbol = Symbol(r"{r}^x", real=True, negative=False)
rz: Symbol = Symbol(r"{r}^z", real=True)
rvechat: Symbol = Symbol(r"\mathbf{\hat{r}}", real=True)
rhat: Symbol = Symbol(r"\hat{r}", real=True, negative=False)
# xhat: Symbol = Symbol(r"\hat{x}", real=True, negative=False)
rxhat: Symbol = Symbol(r"\hat{r}^x", real=True, negative=False)
rzhat: Symbol = Symbol(r"\hat{r}^z", real=True)
rdot_vec: Symbol = MatrixSymbol(r"v", 2, 1)
rdot: Symbol = Symbol(r"v", real=True)
rdotx: Symbol = Symbol(r"v^x", real=True, positive=True)
rdotz: Symbol = Symbol(r"v^z", real=True)
rdotxhat: Symbol = Symbol(r"\hat{v}^x", real=True, positive=True)
rdotzhat: Symbol = Symbol(r"\hat{v}^z", real=True)
rdotx_true: Symbol = Symbol(r"\dot{r}^x", real=True, positive=True)
rdotz_true: Symbol = Symbol(r"\dot{r}^z", real=True)
vdotx: Symbol = Symbol(r"\dot{v}^x", real=True)
vdotz: Symbol = Symbol(r"\dot{v}^z", real=True)
pdotx: Symbol = Symbol(r"\dot{p}_x", real=True)
pdotz: Symbol = Symbol(r"\dot{p}_z", real=True)
ta: Symbol = Symbol(r"a", real=True, positive=True)
tb: Symbol = Symbol(r"b", real=True, positive=True)

x: Symbol = Symbol(r"x", real=True, negative=False)
A: Symbol = Symbol(r"A", real=True, negative=False)
S: Symbol = Symbol(r"S", real=True, negative=False)
y: Symbol = Symbol(r"y", real=True, negative=False)
Delta_x: Symbol = Symbol(r"\Delta{x}", real=True, positive=True)
xx: Symbol = Symbol(r"\tilde{x}", real=True)
z: Symbol = Symbol(r"z", real=True)
x_h: Symbol = Symbol(r"x_h", real=True, positive=True)
x_sigma: Symbol = Symbol(r"x_{\sigma}", real=True, positive=True)

h: Symbol = Symbol(r"h", real=True)
hx: Symbol = Symbol(r"h^x", real=True, negative=False)
hz: Symbol = Symbol(r"h^z", real=True)
h_fn: Function = Function(r"h", real=True, positive=True)
h_0: Symbol = Symbol(r"h_0", real=True, positive=True)
h_0p9: Symbol = Symbol(r"h_{0.9}", real=True)

theta: Symbol = Symbol(r"\theta", real=True, positive=True)
kappa_h: Symbol = Symbol(r"\kappa_\mathrm{h}", real=True, positive=True)

u_0: Symbol = Symbol(r"u_0", real=True, positive=True)
Lc: Symbol = Symbol(r"L_\mathrm{c}", real=True, positive=True)
xi: Symbol = Symbol(r"\xi^{\perp}", real=True, positive=True)
xiv: Symbol = Symbol(r"\xi^{\downarrow}", real=True)
xivhat: Symbol = Symbol(r"\hat{\xi}^{\downarrow}", real=True)
xivhat_0: Symbol = Symbol(r"\hat{\xi}^{\downarrow_0}", real=True)
xiv_0: Symbol = Symbol(r"\xi^{\downarrow_{0}}", real=True, positive=True)
xih: Symbol = Symbol(r"\xi^{\rightarrow}", real=True)
xihhat: Symbol = Symbol(r"\hat{\xi}^{\rightarrow}", real=True)
xih_0: Symbol = Symbol(r"\xi^{\rightarrow_{0}}", real=True, positive=True)
xiv_0_sqrd: Symbol = Symbol(r"\xi^{\downarrow_{0}}^2", real=True)

varphi_rx: Function = Function(r"\varphi", real=True, positive=True)
varphi_rxhat_fn: Function = Function(r"\varphi", real=True, positive=True)
d_varphi_rx: Function = Function(r"\varphi^{\prime}", real=True)
varphi_r: Function = Function(r"\varphi", real=True, positive=True)
varphi_rhat: Function = Function(r"\varphi", real=True, positive=True)
d_varphi: Symbol = Symbol(r"\varphi^{\prime}", real=True)
varphi_0: Symbol = Symbol(r"\varphi_0", real=True, positive=True)
varphi: Symbol = Symbol(r"\varphi", real=True, positive=True)
varphi_c: Symbol = Symbol(r"\varphi_{\mathrm{c}}", real=True, positive=True)
chi_0: Symbol = Symbol(r"\chi_0", real=True, positive=True)
chi: Symbol = Symbol(r"\chi", real=True, positive=True)
varepsilon: Symbol = Symbol(r"\varepsilon", real=True, positive=True)
varepsilonhat: Symbol = Symbol(r"\hat{\varepsilon}", real=True, positive=True)

J: MatrixSymbol = MatrixSymbol(r"J", 2, 2)
g: MatrixSymbol = MatrixSymbol(r"g", 2, 2)
gstar: MatrixSymbol = MatrixSymbol(r"g_*", 2, 2)
gstarhat: MatrixSymbol = MatrixSymbol(r"\hat{g}_*", 2, 2)
det_gstar: Symbol = Symbol(r"{\det}\left(g_*\right)")
pcovec_wrong: MatrixSymbol = MatrixSymbol(r"\mathbf{\widetilde{p}}", 2, 1)
pcovec: MatrixSymbol = MatrixSymbol(r"\mathbf{\widetilde{p}}", 1, 2)
pdotcovec: MatrixSymbol = MatrixSymbol(r"\mathbf{\dot{\widetilde{p}}}", 1, 2)
rdotvec: MatrixSymbol = MatrixSymbol(r"\mathbf{v}", 2, 1)
detJ: Symbol = Symbol(r"det(J)", real=True, positive=True)

eta: Symbol = Symbol(r"\eta", real=True, negative=False)
lmbda: Symbol = Symbol(r"\lambda", real=True, positive=True)
kappa: Symbol = Symbol(r"\kappa", real=True, positive=True)

xhat: Symbol = Symbol(r"\hat{x}", real=True, negative=False)
dzdx: Symbol = Symbol(
    r"\frac{\mathrm{d}{\hat{z}}}{\mathrm{d}{\hat{x}}}", real=True
)

drxdrz: Symbol = Symbol(r"\frac{\mathrm{d}{r^x}}{\mathrm{d}{r^z}}", real=True)
drzdrx: Symbol = Symbol(r"\frac{\mathrm{d}{r^z}}{\mathrm{d}{r^x}}", real=True)
dpxdpz: Symbol = Symbol(r"\frac{\mathrm{d}{p_x}}{\mathrm{d}{p_z}}", real=True)


alpha_tfn: Function = Function(r"\alpha", real=True)  # (t)
beta_tfn: Function = Function(r"\beta", real=True)  # (t)
rx_tfn: Function = Function(r"{r}^x", real=True, positive=True)  # (t)
rz_tfn: Function = Function(r"{r}^z", real=True)  # (t)
px_tfn: Function = Function(r"{p}_x", real=True, positive=True)  # (t)
pz_tfn: Function = Function(r"{p}_z", real=True, negative=True)  # (t)
rdotx_tfn: Function = Function(r"v^x", real=True)  # (t)
rdotz_tfn: Function = Function(r"v^z", real=True)  # (t)
pdotx_tfn: Function = Function(r"\dot{p}_x", real=True)  # (t)
pdotz_tfn: Function = Function(r"\dot{p}_z", real=True)  # (t)
rdotxhat_thatfn: Function = Function(r"\hat{v}^x", real=True)  # (that)
rdotzhat_thatfn: Function = Function(r"\hat{v}^z", real=True)  # (that)
pdotxhat_thatfn: Function = Function(r"\dot{\hat{p}}_x", real=True)  # (that)
pdotzhat_thatfn: Function = Function(r"\dot{\hat{p}}_z", real=True)  # (that)

astar_riem: Symbol = Symbol(r"\alpha^*_{\mathrm{Kr}}", real=True)
bstar_vec: Symbol = Symbol(r"\beta^*_{\mathrm{Kr}}", real=True)

# Channel incision number
Ci: Symbol = Symbol(r"\mathsf{Ci}", real=True, negative=False)

SI.set_quantity_dimension(h_0p9, length)
SI.set_quantity_dimension(xiv_0, length / time)
SI.set_quantity_dimension(xih_0, length / time)
SI.set_quantity_dimension(Ci, 1)
SI.set_quantity_dimension(Lc, length)
SI.set_quantity_dimension(varphi_0, length / time)
SI.set_quantity_dimension(th_0, time)
SI.set_quantity_dimension(th_0p9, time)
SI.set_quantity_dimension(th_0p95, time)
SI.set_quantity_dimension(tv_0, time)
# SI.set_quantity_dimension(t_Lc, time)
SI.set_quantity_dimension(t_oneyear, time)
SI.set_quantity_dimension(t_My, time)
SI.set_quantity_dimension(varphi_c, length / time)


s: Symbol = Symbol(r"s", real=True)
vx: Symbol = Symbol(r"x", real=True)
vz: Symbol = Symbol(r"z", real=True)
v: Symbol = Symbol(r"v", real=True, positive=True)


# Hillslope additions
px_h: Symbol = Symbol(r"p_x", real=True, positive=True)
pz_h: Symbol = Symbol(r"p_z", real=True, negative=True)

xi_h0: Symbol = Symbol(r"\xi^{\perp_0}", real=True, positive=True)
xi_h: Function = Function(r"\xi^{\perp}", real=True, positive=True)

n_h: Symbol = Symbol(r"n", int=True, positive=True)

phi: Symbol = Symbol(r"\phi", real=True, positive=True)
beta_h: Symbol = Symbol(r"\beta", real=True, positive=True)

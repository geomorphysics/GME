"""
---------------------------------------------------------------------

Mathematical symbols used in GME equations,
and their latex representation for pretty printing in notebooks.


---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`SymPy <sympy>`

---------------------------------------------------------------------

"""
from sympy.physics.units.systems import SI
from sympy.physics.units import length, time
from sympy import symbols, MatrixSymbol, Function, Symbol

# import logging
# logging.info('gme.core.symbols')


i: Symbol = symbols(r"i", real=True)
j: Symbol = symbols(r"j", real=True)
k: Symbol = symbols(r"k", real=True)
l: Symbol = symbols(r"l", real=True)
m: Symbol = symbols(r"m", real=True)
F: Symbol = symbols(r"\mathcal{F}", real=True, positive=True)
Fstar: Symbol = symbols(r"\mathcal{F}_*", real=True, positive=True)
Fstar_px_pz: Symbol = symbols(
    r"\mathcal{F}_{*}(p_x\,p_z)", real=True, positive=True
)
H: Symbol = symbols(r"\mathcal{H}", real=True, negative=False)
# ideally, both should be positive=True
L: Symbol = symbols(r"\mathcal{L}", real=True, negative=False)
G1: Symbol = symbols(r"G1")
G2: Symbol = symbols(r"G2")
p: Symbol = symbols(r"p", real=True, positive=True)
p_0: Symbol = symbols(r"p_0", real=True, positive=True)
u: Symbol = symbols(r"u", real=True)
px: Symbol = symbols(r"p_x", real=True)
pz: Symbol = symbols(r"p_z", real=True, negative=True)
pxhat: Symbol = symbols(r"\hat{p}_x", real=True)
pzhat_0: Symbol = symbols(r"\hat{p}_{z_0}", real=True, negative=True)
pzhat: Symbol = symbols(r"\hat{p}_z", real=True, negative=True)
# pz_0: Symbol = symbols(r"p_{z_0}", real=True)
px_min: Symbol = symbols(r"p_{x_\text{min}}", real=True)
pz_min: Symbol = symbols(r"p_{z_\text{min}}", real=True)
beta_max: Symbol = symbols(r"\beta_{\text{max}}")
alpha_max: Symbol = symbols(r"\alpha_{\text{max}}", real=True)
pxp: Symbol = symbols(r"p_x^+", real=True, positive=True)
pzp: Symbol = symbols(r"p_z^+", real=True, positive=True)
pzm: Symbol = symbols(r"p_z^-", real=True, negative=True)
pz_0: Symbol = symbols(r"p_{z_0}", real=True, negative=True)
px_0: Symbol = symbols(r"p_{x_0}", real=True, positive=True)
c: Symbol = symbols(r"c", real=True)

psqrd_substn: Symbol = px ** 2 + pz ** 2
# eta: Symbol = symbols(r"\eta", real=True, positive=True)
mu: Symbol = symbols(r"\mu", real=True, positive=True)
epsilon: Symbol = symbols(r"\epsilon", real=True, positive=True)

t: Symbol = symbols(r"t", real=True, negative=False)
# t_Lc: Symbol = symbols(r't^{\rightarrow\mathrm{L_c}}',
# real=True, positive=True)
th_0p95: Symbol = symbols(r"t^{\rightarrow{0.95}}", real=True, negative=False)
h_0p95: Symbol = symbols(r"h_{0.95}", real=True, negative=False)
th_0p9: Symbol = symbols(r"t^{\rightarrow{0.9}}", real=True, negative=False)
# h_0p9: Symbol = symbols(r"h_{0.9}", real=True, negative=False)
t_oneyear: Symbol = symbols(r"t_{\mathrm{1y}}", real=True, positive=True)
t_My: Symbol = symbols(r"t_{\mathrm{My}}", real=True, positive=True)
that: Symbol = symbols(r"\hat{t}", real=True, negative=False)
tv_0: Symbol = symbols(r"t^{\downarrow_{0}}", real=True, positive=True)
th_0: Symbol = symbols(r"t^{\rightarrow_{0}}", real=True, positive=True)

beta: Symbol = symbols(r"\beta", real=True)
beta_: Symbol = symbols(r"\beta_x", real=True)
beta_crit: Symbol = symbols(r"\beta_c", real=True)
beta_0: Symbol = symbols(r"\beta_0", real=True, positive=True)
SI.set_quantity_dimension(beta_0, 1)
betaplus: Symbol = symbols(r"\beta^+", real=True, positive=True)
alpha: Symbol = symbols(r"\alpha", real=True)
alpha_ext: Symbol = symbols(r"\alpha_\mathrm{ext}", real=True)
alphaplus: Symbol = symbols(r"\alpha^+", real=True, positive=True)
alpha_extremum: Symbol = symbols(r"\alpha_\text{extremum}")
beta_at_alpha_extremum: Symbol = symbols(r"\beta_{\text{extremum}\{\alpha\}}")
phi: Symbol = symbols(r"\phi", real=True)
psi: Symbol = symbols(r"\psi", real=True)

rvec: Symbol = symbols(r"\mathbf{r}", real=True)
r: Symbol = symbols(r"r", real=True, negative=False)
rx: Symbol = symbols(r"{r}^x", real=True, negative=False)
rz: Symbol = symbols(r"{r}^z", real=True)
rvechat: Symbol = symbols(r"\mathbf{\hat{r}}", real=True)
rhat: Symbol = symbols(r"\hat{r}", real=True, negative=False)
# xhat: Symbol = symbols(r"\hat{x}", real=True, negative=False)
rxhat: Symbol = symbols(r"\hat{r}^x", real=True, negative=False)
rzhat: Symbol = symbols(r"\hat{r}^z", real=True)
rdot_vec: Symbol = MatrixSymbol(r"v", 2, 1)
rdot: Symbol = symbols(r"v", real=True)
rdotx: Symbol = symbols(r"v^x", real=True, positive=True)
rdotz: Symbol = symbols(r"v^z", real=True)
rdotxhat: Symbol = symbols(r"\hat{v}^x", real=True, positive=True)
rdotzhat: Symbol = symbols(r"\hat{v}^z", real=True)
rdotx_true: Symbol = symbols(r"\dot{r}^x", real=True, positive=True)
rdotz_true: Symbol = symbols(r"\dot{r}^z", real=True)
vdotx: Symbol = symbols(r"\dot{v}^x", real=True)
vdotz: Symbol = symbols(r"\dot{v}^z", real=True)
pdotx: Symbol = symbols(r"\dot{p}_x", real=True)
pdotz: Symbol = symbols(r"\dot{p}_z", real=True)
ta: Symbol = symbols(r"a", real=True, positive=True)
tb: Symbol = symbols(r"b", real=True, positive=True)

x: Symbol = symbols(r"x", real=True, negative=False)
A: Symbol = symbols(r"A", real=True, negative=False)
S: Symbol = symbols(r"S", real=True, negative=False)
y: Symbol = symbols(r"y", real=True, negative=False)
Delta_x: Symbol = symbols(r"\Delta{x}", real=True, positive=True)
xx: Symbol = symbols(r"\tilde{x}", real=True)
z: Symbol = symbols(r"z", real=True)
x_h: Symbol = symbols(r"x_h", real=True, positive=True)
x_sigma: Symbol = symbols(r"x_{\sigma}", real=True, positive=True)

h: Symbol = symbols(r"h", real=True)
hx: Symbol = symbols(r"h^x", real=True, negative=False)
hz: Symbol = symbols(r"h^z", real=True)
h_fn: Symbol = Function(r"h", real=True, positive=True)
h_0: Symbol = symbols(r"h_0", real=True, positive=True)
h_0p9: Symbol = symbols(r"h_{0.9}", real=True)

theta: Symbol = symbols(r"\theta", real=True, positive=True)
kappa_h: Symbol = symbols(r"\kappa_\mathrm{h}", real=True, positive=True)

u_0: Symbol = symbols(r"u_0", real=True, positive=True)
Lc: Symbol = symbols(r"L_\mathrm{c}", real=True, positive=True)
xi: Symbol = symbols(r"\xi^{\perp}", real=True, positive=True)
xiv: Symbol = symbols(r"\xi^{\downarrow}", real=True)
xivhat: Symbol = symbols(r"\hat{\xi}^{\downarrow}", real=True)
xivhat_0: Symbol = symbols(r"\hat{\xi}^{\downarrow_0}", real=True)
xiv_0: Symbol = symbols(
    r"\xi^{\downarrow_{0}}", real=True, positive=True
)  # NEW: positive=True
xih: Symbol = symbols(r"\xi^{\rightarrow}", real=True)
xihhat: Symbol = symbols(r"\hat{\xi}^{\rightarrow}", real=True)
xih_0: Symbol = symbols(
    r"\xi^{\rightarrow_{0}}", real=True, positive=True
)  # NEW: positive=True
xiv_0_sqrd: Symbol = symbols(r"\xi^{\downarrow_{0}}^2", real=True)

varphi_rx: Function = Function(r"\varphi", real=True, positive=True)
varphi_rxhat_fn: Function = Function(r"\varphi", real=True, positive=True)
d_varphi_rx: Function = Function(r"\varphi^{\prime}", real=True)
varphi_r: Function = Function(r"\varphi", real=True, positive=True)
varphi_rhat: Function = Function(r"\varphi", real=True, positive=True)
d_varphi: Symbol = symbols(r"\varphi^{\prime}", real=True)
varphi_0: Symbol = symbols(r"\varphi_0", real=True, positive=True)
varphi: Symbol = symbols(r"\varphi", real=True, positive=True)
varphi_c: Symbol = symbols(r"\varphi_{\mathrm{c}}", real=True, positive=True)
chi_0: Symbol = symbols(r"\chi_0", real=True, positive=True)
chi: Symbol = symbols(r"\chi", real=True, positive=True)
varepsilon: Symbol = symbols(r"\varepsilon", real=True, positive=True)
varepsilonhat: Symbol = symbols(r"\hat{\varepsilon}", real=True, positive=True)

J: MatrixSymbol = MatrixSymbol(r"J", 2, 2)
g: MatrixSymbol = MatrixSymbol(r"g", 2, 2)
gstar: MatrixSymbol = MatrixSymbol(r"g_*", 2, 2)
gstarhat: MatrixSymbol = MatrixSymbol(r"\hat{g}_*", 2, 2)
det_gstar: Symbol = symbols(r"{\det}\left(g_*\right)")
pcovec_wrong: MatrixSymbol = MatrixSymbol(r"\mathbf{\widetilde{p}}", 2, 1)
pcovec: MatrixSymbol = MatrixSymbol(r"\mathbf{\widetilde{p}}", 1, 2)
pdotcovec: MatrixSymbol = MatrixSymbol(r"\mathbf{\dot{\widetilde{p}}}", 1, 2)
rdotvec: MatrixSymbol = MatrixSymbol(r"\mathbf{v}", 2, 1)
detJ: Symbol = symbols(r"det(J)", real=True, positive=True)

eta: Symbol = symbols(r"\eta", real=True, negative=False)
lmbda: Symbol = symbols(r"\lambda", real=True, positive=True)
kappa: Symbol = symbols(r"\kappa", real=True, positive=True)

xhat: Symbol = symbols(r"\hat{x}", real=True, negative=False)
dzdx: Symbol = symbols(
    r"\frac{\mathrm{d}{\hat{z}}}{\mathrm{d}{\hat{x}}}", real=True
)

drxdrz: Symbol = symbols(r"\frac{\mathrm{d}{r^x}}{\mathrm{d}{r^z}}", real=True)
drzdrx: Symbol = symbols(r"\frac{\mathrm{d}{r^z}}{\mathrm{d}{r^x}}", real=True)
dpxdpz: Symbol = symbols(r"\frac{\mathrm{d}{p_x}}{\mathrm{d}{p_z}}", real=True)


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

astar_riem: Symbol = symbols(r"\alpha^*_{\mathrm{Kr}}", real=True)
bstar_1form: Symbol = symbols(r"\beta^*_{\mathrm{Kr}}", real=True)

# Channel incision number
Ci: Symbol = symbols(r"\mathsf{Ci}", real=True, negative=False)

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

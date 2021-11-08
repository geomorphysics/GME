"""
---------------------------------------------------------------------

Mathematical symbols used in GME equations,
and their latex representation for pretty printing in notebooks.

This module is imported wholesale by most of the other GME modules (using `import *`)
which is generally a _terrible_ idea (because it blindly pollutes a module's entire namespace
and risks a naming clash with other variables in the module), but is necessary for brevity when
manipulating SymPy equations.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`sympy`

---------------------------------------------------------------------

"""

import sympy as sy
from sympy import symbols, MatrixSymbol, Function
# Import units & dimensions
from sympy.physics.units \
    import Quantity, Dimension, \
            length, speed, velocity, time, \
            km, cm, mm, meter, meters, \
            convert_to, percent, degrees, radians
from sympy.physics.units.systems import SI

i,j,k, l, m = symbols(r'i j k, l, m',  real=True)
F = symbols(r'\mathcal{F}',  real=True, positive=True)
Fstar = symbols(r'\mathcal{F}_*',  real=True, positive=True)
Fstar_px_pz = symbols(r'\mathcal{F}_{*}(p_x\,p_z)',  real=True, positive=True)
H = symbols(r'\mathcal{H}',  real=True, negative=False)
L = symbols(r'\mathcal{L}',  real=True, negative=False) # ideally, both should be positive=True
G1 = symbols(r'G1')
G2 = symbols(r'G2')
p = symbols(r'p',  real=True, positive=True)
p_0 = symbols(r'p_0',  real=True, positive=True)
u = symbols(r'u',  real=True)
px = symbols(r'p_x',  real=True)
pz = symbols(r'p_z',  real=True, negative=True)
pxhat = symbols(r'\hat{p}_x',  real=True)
pzhat = symbols(r'\hat{p}_z',  real=True, negative=True)
pz_0 = symbols(r'p_{z_0}',  real=True)
px_min = symbols(r'p_{x_\text{min}}',  real=True)
pz_min = symbols(r'p_{z_\text{min}}',  real=True)
beta_max = symbols(r'\beta_{\text{max}}')
alpha_max = symbols(r'\alpha_{\text{max}}',  real=True)
pxp = symbols(r'p_x^+',  real=True, positive=True)
pzp = symbols(r'p_z^+',  real=True, positive=True)
pzm = symbols(r'p_z^-',  real=True, negative=True)
pz_0 = symbols(r'p_{z_0}',  real=True, negative=True)
px_0 = symbols(r'p_{x_0}',  real=True, positive=True)
c = symbols(r'c', real=True)

psqrd_substn = px**2+pz**2
eta = symbols(r'\eta',  real=True, positive=True)
mu = symbols(r'\mu',  real=True, positive=True)
epsilon = symbols(r'\epsilon',  real=True, positive=True)

t = symbols(r't',  real=True, negative=False)  # New: negative=False
# t_Lc = symbols(r't^{\rightarrow\mathrm{L_c}}',  real=True, positive=True)
th_0p95 = symbols(r't^{\rightarrow{0.95}}',  real=True, negative=False)
h_0p95 = symbols(r'h_{0.95}',  real=True, negative=False)
th_0p9 = symbols(r't^{\rightarrow{0.9}}',  real=True, negative=False)
h_0p9 = symbols(r'h_{0.9}',  real=True, negative=False)
t_oneyear = symbols(r't_{\mathrm{1y}}',  real=True, positive=True)
t_My      = symbols(r't_{\mathrm{My}}',  real=True, positive=True)
that = symbols(r'\hat{t}',  real=True, negative=False)  # New: negative=False
tv_0 = symbols(r't^{\downarrow_{0}}',  real=True, positive=True)
th_0 = symbols(r't^{\rightarrow_{0}}',  real=True, positive=True)

beta = symbols(r'\beta',  real=True)
beta_ = symbols(r'\beta_x',  real=True)
beta_crit = symbols(r'\beta_c',  real=True)
beta_0 = symbols(r'\beta_0',  real=True, positive=True)  # NEW: positive=True
SI.set_quantity_dimension(beta_0, 1)
betaplus = symbols(r'\beta^+',  real=True, positive=True)
alpha = symbols(r'\alpha',  real=True)
alpha_crit = symbols(r'\alpha_c',  real=True)
alphaplus = symbols(r'\alpha^+',  real=True, positive=True)
alpha_extremum = symbols(r'\alpha_\text{extremum}')
beta_at_alpha_extremum = symbols(r'\beta_{\text{extremum}\{\alpha\}}')
phi = symbols(r'\phi',  real=True)

rvec = symbols(r'\mathbf{r}',  real=True)
r = symbols(r'r',  real=True, negative=False)
rx = symbols(r'{r}^x',  real=True, negative=False)
rz = symbols(r'{r}^z',  real=True)
rvechat = symbols(r'\mathbf{\hat{r}}',  real=True)
rhat = symbols(r'\hat{r}',  real=True, negative=False)
xhat = symbols(r'\hat{x}',  real=True, negative=False)
rxhat = symbols(r'\hat{r}^x',  real=True, negative=False)
rzhat = symbols(r'\hat{r}^z',  real=True)
rdot_vec =  MatrixSymbol(r'v',2,1)
rdot = symbols(r'v',  real=True)
rdotx = symbols(r'v^x',  real=True, positive=True)
rdotz = symbols(r'v^z',  real=True)
rdotxhat = symbols(r'\hat{v}^x',  real=True, positive=True)
rdotzhat = symbols(r'\hat{v}^z',  real=True)
rdotx_true = symbols(r'\dot{r}^x',  real=True, positive=True)
rdotz_true = symbols(r'\dot{r}^z',  real=True)
vdotx, vdotz = symbols(r'\dot{v}^x \dot{v}^z',  real=True)
pdotx, pdotz = symbols(r'\dot{p}_x \dot{p}_z',  real=True)
ta = symbols(r'a',  real=True, positive=True)
tb = symbols(r'b',  real=True, positive=True)

x = symbols(r'x',  real=True, negative=False)
A = symbols(r'A',  real=True, negative=False)
S = symbols(r'S',  real=True, negative=False)
y = symbols(r'y',  real=True, negative=False)
Delta_x = symbols(r'\Delta{x}',  real=True, positive=True)
xx = symbols(r'\tilde{x}',  real=True)
z = symbols(r'z',  real=True)
x_h = symbols(r'x_h',  real=True, positive=True)
x_sigma = symbols(r'x_{\sigma}',  real=True, positive=True)

h = symbols(r'h',  real=True)
hx = symbols(r'h^x',  real=True, negative=False)
hz = symbols(r'h^z',  real=True)
h_fn = Function(r'h', real=True, positive=True)(x)
h_0 = symbols(r'h_0',  real=True, positive=True)
h_0p9 = symbols(r'h_{0.9}',  real=True)

theta = symbols(r'\theta',  real=True, positive=True)
kappa_h = symbols(r'\kappa_\mathrm{h}',  real=True, positive=True)

u_0 = symbols(r'u_0',  real=True, positive=True)
x_1 = symbols(r'L_\mathrm{c}',  real=True, positive=True)
Lc = x_1
xi = symbols(r'\xi^{\perp}',  real=True, positive=True)
xiv = symbols(r'\xi^{\downarrow}',  real=True)
xivhat = symbols(r'\hat{\xi}^{\downarrow}',  real=True)
xiv_0 = symbols(r'\xi^{\downarrow_{0}}',  real=True, positive=True)   # NEW: positive=True
xih = symbols(r'\xi^{\rightarrow}',  real=True)
xihhat = symbols(r'\hat{\xi}^{\rightarrow}',  real=True)
xih_0 = symbols(r'\xi^{\rightarrow_{0}}',  real=True, positive=True)   # NEW: positive=True
xiv_0_sqrd = symbols(r'\xi^{\downarrow_{0}}^2',  real=True)

varphi_rx = Function(r'\varphi', real=True, positive=True)(rx)
varphi_rxhat = Function(r'\varphi', real=True, positive=True)(rxhat)
d_varphi_rx = Function(r'\varphi^{\prime}', real=True)(rx)
d_varphi = symbols(r'\varphi^{\prime}', real=True)
varphi_r = Function(r'\varphi', real=True, positive=True)(rvec)
varphi_rhat = Function(r'\varphi', real=True, positive=True)(rvechat)
varphi_0, varphi = symbols(r'\varphi_0 \varphi',  real=True, positive=True)
chi_0, chi       = symbols(r'\chi_0 \chi',  real=True, positive=True)
varepsilon       = symbols(r'\varepsilon',  real=True, positive=True)
varepsilonhat    = symbols(r'\hat{\varepsilon}',  real=True, positive=True)

J       = MatrixSymbol(r'J',2,2)
g       = MatrixSymbol(r'g',2,2)
gstar   = MatrixSymbol(r'g_*',2,2)
det_gstar = symbols(r'{\det}\left(g_*\right)')
pcovec_wrong    = MatrixSymbol(r'\mathbf{\widetilde{p}}',2,1)
pcovec    = MatrixSymbol(r'\mathbf{\widetilde{p}}',1,2)
pdotcovec    = MatrixSymbol(r'\mathbf{\dot{\widetilde{p}}}',1,2)
rdotvec = MatrixSymbol(r'\mathbf{v}',2,1)
detJ = symbols(r'det(J)',  real=True, positive=True)

eta = symbols(r'\eta',  real=True, negative=False)
lmbda = symbols(r'\lambda',  real=True, positive=True)
kappa = symbols(r'\kappa',  real=True, positive=True)

xhat = symbols(r'\hat{x}',  real=True, negative=False)
dzdx = symbols(r'\frac{\mathrm{d}{\hat{z}}}{\mathrm{d}{\hat{x}}}', real=True)

drxdrz = symbols(r'\frac{\mathrm{d}{r^x}}{\mathrm{d}{r^z}}', real=True)
drzdrx = symbols(r'\frac{\mathrm{d}{r^z}}{\mathrm{d}{r^x}}', real=True)
dpxdpz = symbols(r'\frac{\mathrm{d}{p_x}}{\mathrm{d}{p_z}}', real=True)


alpha_tfn = Function(r'\alpha', real=True)(t)
beta_tfn = Function(r'\beta', real=True)(t)
rx_tfn = Function(r'{r}^x', real=True, positive=True)(t)
rz_tfn = Function(r'{r}^z', real=True)(t)
px_tfn = Function(r'{p}_x', real=True, positive=True)(t)
pz_tfn = Function(r'{p}_z', real=True, negative=True)(t)
rdotx_tfn = Function(r'v^x', real=True)(t)
rdotz_tfn = Function(r'v^z', real=True)(t)
pdotx_tfn = Function(r'\dot{p}_x', real=True)(t)
pdotz_tfn = Function(r'\dot{p}_z', real=True)(t)
rdotxhat_thatfn = Function(r'\hat{v}^x', real=True)(that)
rdotzhat_thatfn = Function(r'\hat{v}^z', real=True)(that)
pdotxhat_thatfn = Function(r'\dot{\hat{p}}_x', real=True)(that)
pdotzhat_thatfn = Function(r'\dot{\hat{p}}_z', real=True)(that)

astar_riem = symbols(r'\alpha^*_{\mathrm{Kr}}', real=True)
bstar_1form = symbols(r'\beta^*_{\mathrm{Kr}}', real=True)

Ci   = symbols(r'\mathsf{Ci}', real=True, negative=False)                # Channel incision number

SI.set_quantity_dimension(h_0p9, length)
SI.set_quantity_dimension(xiv_0, length/time)
SI.set_quantity_dimension(xih_0, length/time)
SI.set_quantity_dimension(Ci, 1)
SI.set_quantity_dimension(Lc, length)
SI.set_quantity_dimension(varphi_0, length/time)
SI.set_quantity_dimension(th_0, time)
SI.set_quantity_dimension(th_0p9, time)
SI.set_quantity_dimension(th_0p95, time)
SI.set_quantity_dimension(tv_0, time)
# SI.set_quantity_dimension(t_Lc, time)
SI.set_quantity_dimension(t_oneyear, time)
SI.set_quantity_dimension(t_My, time)

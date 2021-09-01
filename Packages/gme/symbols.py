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

i,j,k, l, m = symbols('i j k, l, m',  real=True)
F = symbols('F',  real=True, positive=True)
Fstar = symbols(r'F_*',  real=True, positive=True)
Fstar_px_pz = symbols(r'F_{*}(p_x\,p_z)',  real=True, positive=True)
H = symbols('H',  real=True, negative=False)
L = symbols('L',  real=True, negative=False) # ideally, both should be positive=True
G1 = symbols('G1')
G2 = symbols('G2')
p = symbols('p',  real=True, positive=True)
p_0 = symbols('p_0',  real=True, positive=True)
u = symbols('u',  real=True)
px = symbols('p_x',  real=True)
pz = symbols('p_z',  real=True, negative=True)
pz_0 = symbols(r'p_{z_0}',  real=True)
px_min = symbols(r'p_{x_\text{min}}',  real=True)
pz_min = symbols(r'p_{z_\text{min}}',  real=True)
beta_max = symbols(r'\beta_{\text{max}}')
alpha_max = symbols(r'\alpha_{\text{max}}',  real=True)
pxp = symbols('p_x^+',  real=True, positive=True)
pzp = symbols('p_z^+',  real=True, positive=True)
pzm = symbols('p_z^-',  real=True, negative=True)
pz_0 = symbols(r'p_{z_0}',  real=True, negative=True)
px_0 = symbols(r'p_{x_0}',  real=True, positive=True)
c = symbols('c', real=True)

psqrd_substn = px**2+pz**2
eta = symbols(r'\eta',  real=True, positive=True)
mu = symbols(r'\mu',  real=True, positive=True)
epsilon = symbols(r'\epsilon',  real=True, positive=True)
t = symbols('t',  real=True)
beta = symbols(r'\beta',  real=True)
beta_ = symbols(r'\beta_x',  real=True)
beta_crit = symbols(r'\beta_c',  real=True)
beta0 = symbols(r'\beta_0',  real=True)
betaplus = symbols(r'\beta^+',  real=True, positive=True)
alpha = symbols(r'\alpha',  real=True)
alpha_crit = symbols(r'\alpha_c',  real=True)
alphaplus = symbols(r'\alpha^+',  real=True, positive=True)
alpha_extremum = symbols(r'\alpha_\text{extremum}')
beta_at_alpha_extremum = symbols(r'\beta_{\text{extremum}\{\alpha\}}')
phi = symbols(r'\phi',  real=True)

rvec = symbols(r'\mathbf{r}',  real=True)
r = symbols('r',  real=True, negative=False)
rx = symbols('{r}^x',  real=True, negative=False)
rz = symbols('{r}^z',  real=True)
rdot_vec =  MatrixSymbol('v',2,1)
rdot = symbols('v',  real=True)
rdotx = symbols('v^x',  real=True, positive=True)
rdotz = symbols('v^z',  real=True)
rdotx_true = symbols(r'\dot{r}^x',  real=True, positive=True)
rdotz_true = symbols(r'\dot{r}^z',  real=True)
vdotx, vdotz = symbols(r'\dot{v}^x \dot{v}^z',  real=True)
pdotx, pdotz = symbols(r'\dot{p}_x \dot{p}_z',  real=True)
ta = symbols('a',  real=True, positive=True)
tb = symbols('b',  real=True, positive=True)

x = symbols('x',  real=True, negative=False)
A = symbols('A',  real=True, negative=False)
S = symbols('S',  real=True, negative=False)
y = symbols('y',  real=True, negative=False)
Delta_x = symbols(r'\Delta{x}',  real=True, positive=True)
xx = symbols(r'\tilde{x}',  real=True)
z = symbols('z',  real=True)
x_h = symbols('x_h',  real=True, positive=True)
x_sigma = symbols('x_{\sigma}',  real=True, positive=True)

h = symbols('h',  real=True)
hx = symbols('h^x',  real=True, negative=False)
hz = symbols('h^z',  real=True)
h_fn = Function('h', real=True, positive=True)(x)
h_0 = symbols('h_0',  real=True, positive=True)
theta = symbols(r'\theta',  real=True, positive=True)
kappa_h = symbols(r'\kappa_\mathrm{h}',  real=True, positive=True)


u_0 = symbols('u_0',  real=True, positive=True)
x_1 = symbols('x_1',  real=True, positive=True)
xi = symbols(r'\xi^{\perp}',  real=True, positive=True)
xiv = symbols(r'\xi^{\downarrow}',  real=True)
xiv_0 = symbols(r'\xi^{\downarrow{0}}',  real=True)
xiv_0_sqrd = symbols(r'\xi^{\downarrow{0}}^2',  real=True)

varphi_rx = Function(r'\varphi', real=True, positive=True)(rx)
d_varphi_rx = Function(r'\varphi\'', real=True, positive=True)(rx)
varphi_r = Function(r'\varphi', real=True, positive=True)(rvec)
varphi_0, varphi = symbols(r'\varphi_0 \varphi',  real=True, positive=True)
chi_0, chi       = symbols(r'\chi_0 \chi',  real=True, positive=True)
varepsilon       = symbols(r'\varepsilon',  real=True, positive=True)

J       = MatrixSymbol('J',2,2)
g       = MatrixSymbol('g',2,2)
gstar   = MatrixSymbol('g_*',2,2)
det_gstar = symbols(r'\det\left(g_*\right)')
pcovec_wrong    = MatrixSymbol(r'\mathbf{\widetilde{p}}',2,1)
pcovec    = MatrixSymbol(r'\mathbf{\widetilde{p}}',1,2)
pdotcovec    = MatrixSymbol(r'\mathbf{\dot{\widetilde{p}}}',1,2)
rdotvec = MatrixSymbol(r'\mathbf{v}',2,1)
detJ = symbols('det(J)',  real=True, positive=True)

eta = symbols(r'\eta',  real=True, negative=False)
lmbda = symbols(r'\lambda',  real=True, positive=True)
kappa = symbols(r'\kappa',  real=True, positive=True)


drxdrz = symbols(r'\frac{\mathrm{d}{r^x}}{\mathrm{d}{r^z}}', real=True)
drzdrx = symbols(r'\frac{\mathrm{d}{r^z}}{\mathrm{d}{r^x}}', real=True)
dpxdpz = symbols(r'\frac{\mathrm{d}{p_x}}{\mathrm{d}{p_z}}', real=True)


alpha_tfn = Function(r'\alpha', real=True)(t)
beta_tfn = Function(r'\beta', real=True)(t)
rx_tfn = Function('{r}^x', real=True, positive=True)(t)
rz_tfn = Function('{r}^z', real=True)(t)
px_tfn = Function('{p}_x', real=True, positive=True)(t)
pz_tfn = Function('{p}_z', real=True, negative=True)(t)
rdotx_tfn = Function('v^x', real=True)(t)
rdotz_tfn = Function('v^z', real=True)(t)
pdotx_tfn = Function(r'\dot{p}_x', real=True)(t)
pdotz_tfn = Function(r'\dot{p}_z', real=True)(t)

astar_riem = symbols(r'\alpha^*_{\mathrm{Kr}}', real=True)
bstar_1form = symbols(r'\beta^*_{\mathrm{Kr}}', real=True)

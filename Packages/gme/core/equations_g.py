"""
---------------------------------------------------------------------

Equation definitions and derivations using :mod:`SymPy <sympy>`.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`sympy`
  -  :mod:`gme`

---------------------------------------------------------------------

"""
# Disable these pylint errors because it doesn't understand SymPy syntax
#   - notably minus signs in equations flag an error
# pylint: disable=invalid-unary-operand-type, not-callable
import warnings

# Typing
# from typing import Dict, Type, Optional  # , Tuple, Any, List

# SymPy
from sympy import Eq, simplify, Matrix, factor, diff

# GME
from gme.core.symbols import \
    eta, gstar, det_gstar, g, varphi_r, rvec

warnings.filterwarnings("ignore")

__all__ = ['EquationsMetricTensorMixin']


class EquationsMetricTensorMixin:
    r"""
    """

    def define_g_eqns(self) -> None:
        r"""
        Define equations for the metric tensor :math:`g` and its dual  :math:`g^*`

        Attributes:
            gstar_varphi_pxpz_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`g^{*} = \left[\begin{matrix}\
                \dfrac{2 p_{x}^{3} \varphi^{2}{\left(\mathbf{r} \right)}}{\left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}}} \
                - \dfrac{3 p_{x}^{3} \left(p_{x}^{2} + \dfrac{3 p_{z}^{2}}{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}}{\left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{5}{2}}} \
                + \dfrac{2 p_{x} \left(p_{x}^{2} + \dfrac{3 p_{z}^{2}}{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}}{\left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}}} \
                & \dfrac{3 p_{x}^{4} p_{z} \varphi^{2}{\left(\mathbf{r} \right)}}{2 \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{5}{2}}} \
                - \dfrac{3 p_{x}^{2} p_{z} \varphi^{2}{\left(\mathbf{r} \right)}}{2 \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}}}\\ \
                \dfrac{3 p_{x}^{2} p_{z} \varphi^{2}{\left(\mathbf{r} \right)}}{\left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}}} \
                - \dfrac{3 p_{x}^{2} p_{z} \left(p_{x}^{2} + \dfrac{3 p_{z}^{2}}{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}}{\left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{5}{2}}} \
                & \dfrac{3 p_{x}^{3} p_{z}^{2} \varphi^{2}{\left(\mathbf{r} \right)}}{2 \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{5}{2}}} \
                - \dfrac{p_{x}^{3} \varphi^{2}{\left(\mathbf{r} \right)}}{2 \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}}}\
                \end{matrix}\right]`
            det_gstar_varphi_pxpz_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`\det\left(g^*\right) \
                = \dfrac{p_{x}^{4} \left(- \dfrac{p_{x}^{2}}{2} \
                + \dfrac{3 p_{z}^{2}}{4}\right) \varphi^{4}{\left(\mathbf{r} \right)}}{p_{x}^{6} + 3 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + p_{z}^{6}}`
            g_varphi_pxpz_eqn (:class:`~sympy.core.relational.Equality`):
                :math:`g = \left[\begin{matrix}\
                \dfrac{2 \left(p_{x}^{2} - 2 p_{z}^{2}\right) \sqrt{p_{x}^{2} + p_{z}^{2}}}{p_{x} \left(2 p_{x}^{2} - 3 p_{z}^{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}} \
                & - \dfrac{6 p_{z}^{3} \sqrt{p_{x}^{2} + p_{z}^{2}}}{p_{x}^{2} \left(2 p_{x}^{2} - 3 p_{z}^{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}}\\ \
                - \dfrac{6 p_{z}^{3} \sqrt{p_{x}^{2} + p_{z}^{2}}}{p_{x}^{2} \left(2 p_{x}^{2} - 3 p_{z}^{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}} \
                & - \dfrac{4 p_{x}^{6} + 14 p_{x}^{4} p_{z}^{2} + 22 p_{x}^{2} p_{z}^{4} + 12 p_{z}^{6}}{p_{x}^{3} \sqrt{p_{x}^{2} + p_{z}^{2}} \left(2 p_{x}^{2} - 3 p_{z}^{2}\right) \varphi^{2}{\left(\mathbf{r} \right)}}\
                \end{matrix}\right]`
            gstar_eigen_varphi_pxpz (list of :class:`~sympy.core.expr.Expr`) :
                eigenvalues and eigenvectors of :math:`g^{*}` in one object
            gstar_eigenvalues (:class:`~sympy.matrices.immutable.ImmutableDenseMatrix`) :
                :math:`\left[\begin{matrix}\
                \dfrac{\varphi_0^{2} p_{x} x_{1}^{- 4 \mu} \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)^{2} \left(- 3 \left(p_{x}^{2} + p_{z}^{2}\right) \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}} + \left(p_{x}^{2} + 6 p_{z}^{2}\right) \left(p_{x}^{6} + 3 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + p_{z}^{6}\right)\right)}{4 \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}} \left(p_{x}^{6} + 3 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + p_{z}^{6}\right)}\\\
                \dfrac{\varphi_0^{2} p_{x} x_{1}^{- 4 \mu} \left(\varepsilon x_{1}^{2 \mu} + \left(x_{1} - {r}^x\right)^{2 \mu}\right)^{2} \left(3 \left(p_{x}^{2} + p_{z}^{2}\right) \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}} + \left(p_{x}^{2} + 6 p_{z}^{2}\right) \left(p_{x}^{6} + 3 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + p_{z}^{6}\right)\right)}{4 \left(p_{x}^{2} + p_{z}^{2}\right)^{\frac{3}{2}} \left(p_{x}^{6} + 3 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + p_{z}^{6}\right)}\
                \end{matrix}\right]`
            gstar_eigenvectors (list containing pair of :class:`~sympy.matrices.immutable.ImmutableDenseMatrix`) :
                :math:`\left[\
                \begin{matrix}\dfrac{p_{x} p_{z}^{3} \left(p_{x}^{6} + 2 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + 2 p_{z}^{6} + \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}}\right)}{p_{x}^{10} + 3 p_{x}^{8} p_{z}^{2} + 7 p_{x}^{6} p_{z}^{4} + 11 p_{x}^{4} p_{z}^{6} + p_{x}^{4} \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}} + 10 p_{x}^{2} p_{z}^{8} + p_{x}^{2} p_{z}^{2} \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}} + 4 p_{z}^{10} + 2 p_{z}^{4} \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}}}\\ \
                1 \
                \end{matrix}\
                \right] \\ \
                \left[\
                \begin{matrix}\dfrac{p_{x} p_{z}^{3} \left(p_{x}^{6} + 2 p_{x}^{4} p_{z}^{2} + 3 p_{x}^{2} p_{z}^{4} + 2 p_{z}^{6} - \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}}\right)}{p_{x}^{10} + 3 p_{x}^{8} p_{z}^{2} + 7 p_{x}^{6} p_{z}^{4} + 11 p_{x}^{4} p_{z}^{6} - p_{x}^{4} \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}} + 10 p_{x}^{2} p_{z}^{8} - p_{x}^{2} p_{z}^{2} \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}} + 4 p_{z}^{10} - 2 p_{z}^{4} \sqrt{p_{x}^{12} + 4 p_{x}^{10} p_{z}^{2} + 10 p_{x}^{8} p_{z}^{4} + 20 p_{x}^{6} p_{z}^{6} + 25 p_{x}^{4} p_{z}^{8} + 16 p_{x}^{2} p_{z}^{10} + 4 p_{z}^{12}}}\\ \
                1 \
                \end{matrix}\right]`
        """
        self.gstar_varphi_pxpz_eqn = None
        self.det_gstar_varphi_pxpz_eqn = None
        self.g_varphi_pxpz_eqn = None
        self.gstar_eigen_varphi_pxpz = None
        self.gstar_eigenvalues = None
        self.gstar_eigenvectors = None

        eta_sub = {eta: self.eta_}
        self.gstar_varphi_pxpz_eqn \
            = Eq(gstar, factor(
                    Matrix([diff(self.rdot_vec_eqn.rhs,
                                 self.p_covec_eqn.rhs[0]).T,
                            diff(self.rdot_vec_eqn.rhs,
                                 self.p_covec_eqn.rhs[1]).T])
                )).subs(eta_sub)
        self.det_gstar_varphi_pxpz_eqn \
            = Eq(det_gstar, (simplify(self.gstar_varphi_pxpz_eqn.rhs
                                      .subs(eta_sub).det())))
        if self.eta_ == 1 and self.beta_type == 'sin':
            print(r'Cannot compute all metric tensor $g^{ij}$ equations '
                  + r'for $\sin\beta$ model and $\eta=1$')
            return
        self.g_varphi_pxpz_eqn \
            = Eq(g, simplify(self.gstar_varphi_pxpz_eqn.rhs
                             .subs(eta_sub).inverse()))
        self.gstar_eigen_varphi_pxpz \
            = self.gstar_varphi_pxpz_eqn.rhs.eigenvects()
        self.gstar_eigenvalues = simplify(
            Matrix([self.gstar_eigen_varphi_pxpz[0][0],
                    self.gstar_eigen_varphi_pxpz[1][0]])
            .subs({varphi_r(rvec): self.varphi_rx_eqn.rhs}))
        self.gstar_eigenvectors = (
            [simplify(Matrix(self.gstar_eigen_varphi_pxpz[0][2][0])
                      .subs({varphi_r(rvec): self.varphi_rx_eqn.rhs})),
             simplify(Matrix(self.gstar_eigen_varphi_pxpz[1][2][0])
                      .subs({varphi_r(rvec): self.varphi_rx_eqn.rhs}))])


#

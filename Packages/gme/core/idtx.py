"""
---------------------------------------------------------------------

Equation definitions and derivations using :mod:`SymPy <sympy>`.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`SymPy <sympy>`
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix:
    https://docs.sympy.org/latest/modules/matrices/immutablematrices.html
.. _Equality: https://docs.sympy.org/latest/modules/core.html
    #sympy.core.relational.Equality

---------------------------------------------------------------------

"""
# Disable these pylint errors because it doesn't understand SymPy syntax
#   - notably minus signs in equations flag an error
# pylint: disable=invalid-unary-operand-type, not-callable
import warnings
import logging

# Typing
# from typing import Dict, Type, Optional  # , Tuple, Any, List

# SymPy
from sympy import Eq, Rational, factor, N, re, im, sqrt, solve, \
                  sin, cos, tan, Abs

# GME
from gme.core.symbols import \
    p, rx, rz, px, pz, beta, eta, rvec, varphi, varphi_r

warnings.filterwarnings("ignore")

__all__ = ['IdtxMixin']


class IdtxMixin:
    r"""
    """
    eta_: float
    beta_type: str
    pz_p_beta_eqn: Eq
    p_varphi_beta_eqn: Eq
    gstar_varphi_pxpz_eqn: Eq

    def define_idtx_fgtx_eqns(self) -> None:
        r"""
        Define indicatrix and figuratrix equations

        Attributes:
            pz_cosbeta_varphi_eqn (`Equality`_):
                :math:`p_{z}^{4} = \dfrac{\cos^{4}{\left(\beta \right)}}
                {\varphi^{4} \left(1
                - \cos^{2}{\left(\beta \right)}\right)^{3}}`

            cosbeta_pz_varphi_solns (list):
                :math:`\left[
                -\frac{ \left(  -6 \varphi^{4} p_{z}^{4} \sqrt[3]{27
                \varphi^{8} p_{z}^{8} - 18 \varphi^{4} p_{z}^{4}
                + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12}
                p_{z}^{12}} + 2} - 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4}
                + 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4}
                + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12}
                p_{z}^{12}} + 2\right)^{\frac{2}{3}}
                + 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4}
                p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16}
                - 108 \varphi^{12} p_{z}^{12}} + 2} + 2 \sqrt[3]{2} \right)
                }{6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16}
                p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2}}
                ,\dots \right]`

            fgtx_cosbeta_pz_varphi_eqn (`Equality`_):
                :math:`\cos^{2}{\left(\beta \right)}
                = - \frac{- 6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8}
                p_{z}^{8} - 18 \varphi^{4} p_{z}^{4}
                + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12}
                p_{z}^{12}} + 2} - 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4}
                + 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16}
                - 108 \varphi^{12} p_{z}^{12}} + 2\right)^{\frac{2}{3}}
                + 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16}
                p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2}
                + 2 \sqrt[3]{2}}{6 \varphi^{4} p_{z}^{4} \sqrt[3]{27
                \varphi^{8} p_{z}^{8} - 18 \varphi^{4} p_{z}^{4}
                + \sqrt{729 \varphi^{16} p_{z}^{16} - 108 \varphi^{12}
                p_{z}^{12}} + 2}}`

            fgtx_tanbeta_pz_varphi_eqn (`Equality`_):
                :math:`\tan{\left(\beta \right)}
                = \sqrt{\frac{- 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4}
                + 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16}
                p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}}
                + 2\right)^{\frac{2}{3}}
                + 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4}
                p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16}
                - 108 \varphi^{12} p_{z}^{12}} + 2} + 2 \sqrt[3]{2}}{6
                \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16}
                p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2}
                + 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} - 2^{\frac{2}{3}}
                \left(27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16}
                p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}}
                + 2\right)^{\frac{2}{3}}
                - 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4}
                p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16}
                - 108 \varphi^{12} p_{z}^{12}} + 2} - 2 \sqrt[3]{2}}}`

            fgtx_px_pz_varphi_eqn (`Equality`_):
                :math:`p_{x}
                = - p_{z} \sqrt{\frac{- 12 \sqrt[3]{2} \varphi^{4}
                p_{z}^{4} + 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16}
                p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}}
                + 2\right)^{\frac{2}{3}}
                + 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4}
                p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16}
                - 108 \varphi^{12} p_{z}^{12}} + 2} + 2 \sqrt[3]{2}}{6
                \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16}
                p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}} + 2}
                + 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} - 2^{\frac{2}{3}}
                \left(27 \varphi^{8} p_{z}^{8}
                - 18 \varphi^{4} p_{z}^{4} + \sqrt{729 \varphi^{16}
                p_{z}^{16} - 108 \varphi^{12} p_{z}^{12}}
                + 2\right)^{\frac{2}{3}}
                - 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} - 18 \varphi^{4}
                p_{z}^{4} + \sqrt{729 \varphi^{16} p_{z}^{16}
                - 108 \varphi^{12} p_{z}^{12}} + 2} - 2 \sqrt[3]{2}}}`

            idtx_rdotx_pz_varphi_eqn (`Equality`_):
                :math:`{r}^x = \dfrac{\sqrt{6} \left(- 81
                \cdot 2^{\frac{2}{3}} \varphi^{12} p_{z}^{12} \sqrt[3]{27
                \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4}
                p_{z}^{4} + 2} - 378 \varphi^{12} p_{z}^{12} - 9 \cdot
                2^{\frac{2}{3}} \sqrt{3} \varphi^{10} p_{z}^{10}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} \sqrt[3]{27 \varphi^{8}
                p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27
                \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2}
                - 42 \sqrt{3} \varphi^{10} p_{z}^{10} \sqrt{27
                \varphi^{4} p_{z}^{4} - 4} + 45\sqrt[3]{2}\varphi^{8}p_{z}^{8}
                \left(27 \varphi^{8} p_{z}^{8} + 3\sqrt{3}\varphi^{6}p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4}
                p_{z}^{4} + 2\right)^{\frac{2}{3}} + 96 \cdot 2^{\frac{2}{3}}
                \varphi^{8} p_{z}^{8} \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4}
                p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2}
                + 306 \varphi^{8} p_{z}^{8} + \sqrt[3]{2}
                \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4}\left(27\varphi^{8}p_{z}^{8}
                + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4}
                p_{z}^{4} - 4} -18\varphi^{4}p_{z}^{4} + 2\right)^{\frac{2}{3}}
                + 2 \cdot 2^{\frac{2}{3}} \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} \sqrt[3]{27 \varphi^{8}
                p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27
                \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2}
                + 6 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4}
                p_{z}^{4} - 4} - 20 \sqrt[3]{2} \varphi^{4} p_{z}^{4} \left(27
                \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4}
                + 2\right)^{\frac{2}{3}} - 26 \cdot 2^{\frac{2}{3}}
                \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4}
                p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} - 64 \varphi^{4}
                p_{z}^{4} + 2 \sqrt[3]{2}\left(27\varphi^{8}p_{z}^{8}+3\sqrt{3}
                \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18
                \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}}
                + 2 \cdot 2^{\frac{2}{3}} \sqrt[3]{27 \varphi^{8} p_{z}^{8} +
                3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4}
                - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 4\right)}{72 \varphi^{4}
                p_{z}^{5} \left(\frac{\sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3
                \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4}
                - 4} - 18 \varphi^{4} p_{z}^{4} + 2}}{6 \varphi^{4} p_{z}^{4}
                \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6}
                p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4}
                p_{z}^{4} + 2} + 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} -
                2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3}
                \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} -
                18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} -2\sqrt[3]{27
                \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4}
                - 18 \varphi^{4} p_{z}^{4} + 2}
                - 2 \sqrt[3]{2}}\right)^{\frac{3}{2}} \sqrt[3]{27 \varphi^{8}
                p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4}
                + 2} \left(- 54 \cdot 2^{\frac{2}{3}} \varphi^{12} p_{z}^{12}
                - 6 \cdot 2^{\frac{2}{3}} \sqrt{3} \varphi^{10} p_{z}^{10}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} + 6 \varphi^{8} p_{z}^{8}
                \left(27 \varphi^{8} p_{z}^{8} + 3\sqrt{3}\varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4}
                + 2\right)^{\frac{2}{3}} + 33 \sqrt[3]{2} \varphi^{8} p_{z}^{8}
                \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6}
                p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4}
                p_{z}^{4} + 2} + 78 \cdot 2^{\frac{2}{3}} \varphi^{8} p_{z}^{8}
                + \sqrt[3]{2} \sqrt{3} \varphi^{6} p_{z}^{6}\sqrt{27\varphi^{4}
                p_{z}^{4} - 4} \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3}
                \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} -
                18 \varphi^{4} p_{z}^{4} + 2}
                + 2 \cdot 2^{\frac{2}{3}} \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 12 \varphi^{4} p_{z}^{4}
                \left(27 \varphi^{8} p_{z}^{8} + 3\sqrt{3}\varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4}
                + 2\right)^{\frac{2}{3}} - 18 \sqrt[3]{2} \varphi^{4} p_{z}^{4}
                \sqrt[3]{27\varphi^{8}p_{z}^{8} + 3\sqrt{3}\varphi^{6}p_{z}^{6}
                \sqrt{27\varphi^{4}p_{z}^{4} - 4} - 18\varphi^{4}p_{z}^{4} + 2}
                - 24 \cdot 2^{\frac{2}{3}} \varphi^{4} p_{z}^{4}
                + 2 \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6}
                p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4}
                p_{z}^{4} + 2\right)^{\frac{2}{3}} + 2 \sqrt[3]{2}
                \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6}
                p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} -
                18 \varphi^{4} p_{z}^{4} + 2} + 2\cdot 2^{\frac{2}{3}}\right)}`

            idtx_rdotz_pz_varphi_eqn (`Equality`_):
                :math:`{r}^z = \dfrac{\sqrt{6} \sqrt{\frac{- 12 \sqrt[3]{2}
                \varphi^{4} p_{z}^{4} + 2^{\frac{2}{3}} \left(27 \varphi^{8}
                p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27
                \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4}
                + 2\right)^{\frac{2}{3}} + 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4}
                p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} + 2
                \sqrt[3]{2}}{6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8}
                p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27
                \varphi^{4} p_{z}^{4} - 4} - 18
                \varphi^{4} p_{z}^{4} + 2} + 12 \sqrt[3]{2} \varphi^{4}
                p_{z}^{4} - 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8} +
                3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4}
                p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} +
                2\right)^{\frac{2}{3}} - 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27
                \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2} - 2
                \sqrt[3]{2}}} \left(27 \cdot 2^{\frac{2}{3}} \varphi^{8}
                p_{z}^{8} + 3 \cdot 2^{\frac{2}{3}}
                \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4}
                - 4} - 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4} \sqrt[3]{27
                \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4}
                p_{z}^{4} + 2} - 18 \cdot 2^{\frac{2}{3}} \varphi^{4}
                p_{z}^{4} + 2 \left(27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3}
                \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4} -
                18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}} +
                2 \sqrt[3]{2} \sqrt[3]{27
                \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4}
                p_{z}^{4} + 2} + 2 \cdot 2^{\frac{2}{3}}\right)}
                {72 \varphi^{4} p_{z}^{5} \left(\frac{\sqrt[3]{27 \varphi^{8}
                p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27
                \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2}}
                {6 \varphi^{4} p_{z}^{4} \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                + 3 \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} -
                18 \varphi^{4} p_{z}^{4} + 2} + 12 \sqrt[3]{2}
                \varphi^{4} p_{z}^{4} - 2^{\frac{2}{3}} \left(27
                \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} -
                18 \varphi^{4} p_{z}^{4} + 2\right)^{\frac{2}{3}}
                - 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8} + 3 \sqrt{3}
                \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4} p_{z}^{4} - 4}
                - 18 \varphi^{4} p_{z}^{4} + 2} -
                2 \sqrt[3]{2}}\right)^{\frac{3}{2}} \sqrt[3]{27 \varphi^{8}
                p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4}
                p_{z}^{4} + 2} \left(- 6 \varphi^{4} p_{z}^{4} \sqrt[3]{27
                \varphi^{8} p_{z}^{8} + 3 \sqrt{3} \varphi^{6} p_{z}^{6}
                \sqrt{27 \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4}
                p_{z}^{4} + 2} - 12 \sqrt[3]{2} \varphi^{4} p_{z}^{4}
                + 2^{\frac{2}{3}} \left(27 \varphi^{8} p_{z}^{8}
                + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27
                \varphi^{4} p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4}
                + 2\right)^{\frac{2}{3}} + 2 \sqrt[3]{27 \varphi^{8} p_{z}^{8}
                + 3 \sqrt{3} \varphi^{6} p_{z}^{6} \sqrt{27 \varphi^{4}
                p_{z}^{4} - 4} - 18 \varphi^{4} p_{z}^{4} + 2}
                + 2 \sqrt[3]{2}\right)}`

        """
        logging.info('core.idtx.define_idtx_fgtx_eqns')
        # if self.eta_ == 2:
        #     pz_tanbeta_varphi_eqn = ( self.pz_p_beta_eqn
        #      .subs({p:self.p_varphi_beta_eqn.rhs})
        #      .subs({varphi_r(rvec):varphi})
        #      .subs({cos(beta):sqrt(1/(1+tan(beta)**2))})
        #      .subs({Abs(tan(beta)):tan(beta)})
        #     )
        #   tanbeta_pz_varphi_solns = solve( pz_tanbeta_varphi_eqn, tan(beta) )
        #     tanbeta_pz_varphi_eqn = Eq(tan(beta),
        #  ([soln for soln in tanbeta_pz_varphi_solns
        #               if Abs(im(N(soln.subs({varphi:1,pz:1}))))<1e-20][0]) )
        #     self.fgtx_tanbeta_pz_varphi_eqn = tanbeta_pz_varphi_eqn
        #     self.fgtx_cosbeta_pz_varphi_eqn
        #         = Eq(cos(beta)**2, 1/(1+tanbeta_pz_varphi_eqn.rhs**2))
        # else:
        eta_sub = {eta: self.eta_}
        pz_cosbeta_varphi_tmp_eqn \
            = self.pz_p_beta_eqn \
            .subs({p: self.p_varphi_beta_eqn.rhs}) \
            .subs({varphi_r(rvec): varphi}) \
            .subs(eta_sub) \
            .subs({Abs(tan(beta)): Abs(sin(beta))/Abs(cos(beta))}) \
            .subs({Abs(cos(beta)): cos(beta), Abs(sin(beta)): sin(beta)}) \
            .subs({sin(beta): sqrt(1-cos(beta)**2)})

        pz_cosbeta_varphi_eqn \
            = Eq(pz_cosbeta_varphi_tmp_eqn.lhs**self.eta_,  # __dbldenom
                 pz_cosbeta_varphi_tmp_eqn.rhs**self.eta_)  # __dbldenom

        # New
        # pz_cosbeta_varphi_eqn \
        # = (self.pz_varphi_beta_eqn
        # .subs({Abs(sin(beta)**self.eta_):sin(beta)**self.eta_})
        #                              .subs({varphi_r(rvec):varphi})
        #                             .subs({sin(beta):sqrt(1-cos(beta)**2)}))
        # cosbeta_pz_varphi_soln
        #  = (solve( pz_cosbeta_varphi_eqn, cos(beta)**2 ))[0]

        self.pz_cosbeta_varphi_eqn = pz_cosbeta_varphi_eqn
        self.cosbeta_pz_varphi_solns = None
        self.cosbeta_pz_varphi_soln = None
        self.fgtx_cosbeta_pz_varphi_eqn = None
        self.fgtx_tanbeta_pz_varphi_eqn = None
        self.fgtx_px_pz_varphi_eqn = None
        self.idtx_rdotx_pz_varphi_eqn = None
        self.idtx_rdotz_pz_varphi_eqn = None
        self.cosbeta_pz_varphi_solns \
            = solve(self.pz_cosbeta_varphi_eqn, cos(beta))
        # self.cosbetasqrd_pz_varphi_solns \
        #     = solve(self.pz_cosbeta_varphi_eqn, cos(beta)**2)
        if (self.eta_ == Rational(1, 4) or self.eta_ == Rational(3, 2)) \
                and self.beta_type == 'tan':
            print('Cannot compute all indicatrix equations for '
                  + rf'$\tan\beta$ model and $\eta={self.eta_}$')
            return

        def find_cosbeta_root(sub):
            # logging.info([
            #     soln.subs(sub) for soln in self.cosbeta_pz_varphi_solns
            #     ])
            rtn = [
                soln for soln in self.cosbeta_pz_varphi_solns
                if Abs(im(N(soln.subs(sub)))) < 1e-20
                and (re(N(soln.subs(sub)))) >= 0
                ]
            # logging.info(rtn)
            return rtn

        self.cosbeta_pz_varphi_soln = find_cosbeta_root(
            {varphi: 1, pz: -0.01})
        if self.cosbeta_pz_varphi_soln == []:
            self.cosbeta_pz_varphi_soln = find_cosbeta_root(
                {varphi: 10, pz: -0.5})
        self.fgtx_cosbeta_pz_varphi_eqn \
            = Eq(cos(beta),
                 self.cosbeta_pz_varphi_soln[0])
        self.fgtx_tanbeta_pz_varphi_eqn \
            = Eq(tan(beta),
                 (1/(self.fgtx_cosbeta_pz_varphi_eqn.rhs)-1))
        self.fgtx_px_pz_varphi_eqn \
            = factor(Eq(px, -pz*self.fgtx_tanbeta_pz_varphi_eqn.rhs))
        g_xx = self.gstar_varphi_pxpz_eqn.rhs[0, 0]
        g_zx = self.gstar_varphi_pxpz_eqn.rhs[1, 0]
        g_xz = self.gstar_varphi_pxpz_eqn.rhs[0, 1]
        g_zz = self.gstar_varphi_pxpz_eqn.rhs[1, 1]
        self.idtx_rdotx_pz_varphi_eqn = factor(
            Eq(rx, (g_xx*px+g_xz*pz)
               .subs({px: self.fgtx_px_pz_varphi_eqn.rhs,
                      varphi_r(rvec): varphi}))
        )
        self.idtx_rdotz_pz_varphi_eqn = factor(factor(
            Eq(rz, (g_zx*px+g_zz*pz)
               .subs({px: self.fgtx_px_pz_varphi_eqn.rhs,
                      varphi_r(rvec): varphi}))
        ))


#

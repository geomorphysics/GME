"""
---------------------------------------------------------------------

Equation definitions and derivations using :mod:`SymPy <sympy>`.

---------------------------------------------------------------------

This module provides a derivation of GME theory using :mod:`SymPy <sympy>`
for a 2D slice of a 3D landscape along a channel transect.

Starting from a model equation for the surface-normal erosion rate in terms of
the tilt angle of the topographic surface and the distance downstream
(used to infer the flow component of the model erosion process), we derive
the fundamental function (of a co-Finsler metric space), the corresponding
Hamiltonian,
and the equivalent metric tensor.
The rest of the equations are derived from these core results.

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
import logging

# Typing
# from typing import Dict, Optional

# GME
from gme.core.equations import EquationsMixedIn, Equations
from gme.core.equations_geodesic import EquationsGeodesicMixin
from gme.core.equations_idtx import EquationsIdtxMixin
from gme.core.equations_ibc import EquationsIbcMixin

warnings.filterwarnings("ignore")

__all__ = ['EquationsGeodesic', 'EquationsIdtx', 'EquationsIbc',
           'EquationsSetupOnly']


class EquationsGeodesic(Equations, EquationsGeodesicMixin):
    r"""
    TBD
    """

    def __init__(self, parameters, **kwargs) -> None:
        r"""
        Constructor method
        """
        super().__init__(parameters=parameters, **kwargs)
        logging.info('EquationsGeodesic')

        self.prep_geodesic_eqns(parameters if self.do_raw else None)
        self.define_geodesic_eqns()  # parameters if not do_raw else None)


class EquationsIdtx(Equations, EquationsIdtxMixin):
    r"""
    TBD
    """

    def __init__(self, parameters, **kwargs) -> None:
        r"""
        Constructor method
        """
        super().__init__(parameters=parameters, **kwargs)
        logging.info('EquationsIdtx')

        self.define_idtx_fgtx_eqns()


class EquationsIbc(Equations, EquationsIbcMixin):
    r"""
    TBD
    """

    def __init__(self, parameters, ibc_type: str = 'convex-up', **kwargs) \
            -> None:
        r"""
        Constructor method
        """
        super().__init__(parameters=parameters, **kwargs)
        logging.info('EquationsIbc')
        self.ibc_type = ibc_type

        self.prep_ibc_eqns()
        self.define_ibc_eqns()
        self.set_ibc_eqns()


class EquationsSetupOnly(
            EquationsMixedIn,
            EquationsGeodesicMixin,
            EquationsIdtxMixin,
            EquationsIbcMixin
        ):
    r"""
    TBD
    """

    def __init__(
        self,
        **kwargs
    ) -> None:
        r"""
        Constructor method.
        """
        logging.info('EquationsSetup')

        super().__init__(**kwargs)


#

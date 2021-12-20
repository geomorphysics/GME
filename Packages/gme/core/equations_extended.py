"""
---------------------------------------------------------------------

Equation definitions and derivations using :mod:`SymPy <sympy>`.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`sympy`
  -  `GME`_

.. _GMPLib: https://github.com/geomorphysics/GMPLib
.. _GME: https://github.com/geomorphysics/GME
.. _Matrix: https://docs.sympy.org/latest/modules/matrices\
/immutablematrices.html

---------------------------------------------------------------------

"""
# Disable these pylint errors because it doesn't understand SymPy syntax
#   - notably minus signs in equations flag an error
# pylint: disable=invalid-unary-operand-type, not-callable
import warnings
import logging

# Typing
from typing import Dict

# GME
from gme.core.equations import EquationsMixedIn, Equations
from gme.core.geodesic import GeodesicMixin
from gme.core.idtx import IdtxMixin
from gme.core.ibc import IbcMixin

warnings.filterwarnings("ignore")

__all__ = ['EquationsGeodesic',
           'EquationsIdtx',
           'EquationsIbc',
           'EquationsSetupOnly']


class EquationsGeodesic(Equations, GeodesicMixin):
    r"""
    Generate set of equations including geodesic equations.
    """

    def __init__(
        self,
        parameters: Dict,
        **kwargs
    ) -> None:
        r"""
        Constructor method
        """
        super().__init__(parameters=parameters, **kwargs)
        logging.info('core.equations_extended.EquationsGeodesic')

        self.prep_geodesic_eqns(parameters if self.do_raw else None)
        self.define_geodesic_eqns()  # parameters if not do_raw else None)


class EquationsIdtx(Equations, IdtxMixin):
    r"""
    Generate set of equations including indicatrix/figuratrix equations.
    """

    def __init__(
        self,
        parameters: Dict,
        **kwargs
    ) -> None:
        r"""
        Constructor method
        """
        super().__init__(parameters=parameters, **kwargs)
        logging.info('core.equations_extended.EquationsIdtx')

        self.define_idtx_fgtx_eqns()


class EquationsIbc(Equations, IbcMixin):
    r"""
    Generate set of equations including initial/boundary condition equations.
    """

    def __init__(
        self,
        parameters: Dict,
        ibc_type: str = 'convex-up',
        **kwargs
    ) -> None:
        r"""
        Constructor method
        """
        super().__init__(parameters=parameters, **kwargs)
        logging.info('core.equations_extended.EquationsIbc')
        self.ibc_type = ibc_type

        self.prep_ibc_eqns()
        self.define_ibc_eqns()
        self.set_ibc_eqns()


class EquationsSetupOnly(
            EquationsMixedIn,
            GeodesicMixin,
            IdtxMixin,
            IbcMixin
        ):
    r"""
    Generate methods to perform equation definitions but don't act on them.
    """

    def __init__(
        self,
        ibc_type: str = 'convex-up',
        **kwargs
    ) -> None:
        r"""
        Constructor method.
        """
        logging.info('core.equations_extended.EquationsSetup')
        self.ibc_type = ibc_type

        super().__init__(**kwargs)


#

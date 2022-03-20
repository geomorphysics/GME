"""
---------------------------------------------------------------------

Derive extended sets of equations, including those describing the loci
of the indicatrix and figuratrix, those for handling convex and concave
initial profiles, and the geodesic equations.

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

# Library
import warnings
import logging
from typing import Dict, Optional

# GME
from gme.core.equations import EquationsMixedIn, Equations
from gme.core.geodesic import GeodesicMixin
from gme.core.idtx import IdtxMixin
from gme.core.ibc import IbcMixin

warnings.filterwarnings("ignore")

__all__ = [
    "EquationsGeodesic",
    "EquationsIdtx",
    "EquationsIbc",
    "EquationsIdtxIbc",
    "EquationsSetupOnly",
]


class EquationsGeodesic(Equations, GeodesicMixin):
    r"""
    Generate set of equations including geodesic equations.
    """

    # Definitions
    ibc_type: str

    def __init__(self, parameters: Optional[Dict] = None, **kwargs) -> None:
        r"""
        Constructor method
        """
        super().__init__(parameters=parameters, **kwargs)
        logging.info("gme.core.equations_extended.EquationsGeodesic")

        self.prep_geodesic_eqns(parameters if self.do_raw else None)
        self.define_geodesic_eqns()
        # parameters if not do_raw else None)


class EquationsIdtx(Equations, IdtxMixin):
    r"""
    Generate set of equations including indicatrix/figuratrix equations.
    """

    def __init__(self, parameters: Optional[Dict] = None, **kwargs) -> None:
        r"""
        Constructor method
        """
        super().__init__(parameters=parameters, **kwargs)
        logging.info("gme.core.equations_extended.EquationsIdtx")

        self.define_idtx_fgtx_eqns()


class EquationsIbc(Equations, IbcMixin):
    r"""
    Generate set of equations including initial/boundary condition equations.
    """

    # Definitions
    ibc_type: str

    def __init__(
        self,
        parameters: Optional[Dict] = None,
        ibc_type: str = "convex-up",
        **kwargs
    ) -> None:
        r"""
        Constructor method
        """
        super().__init__(parameters=parameters, **kwargs)
        logging.info("gme.core.equations_extended.EquationsIbc")
        self.ibc_type: str = ibc_type

        self.prep_ibc_eqns()
        self.define_ibc_eqns()
        self.set_ibc_eqns()


class EquationsIdtxIbc(EquationsIdtx, IbcMixin):
    r"""
    Generate set of equations including indicatrix/figuratrix
    and initial/boundary condition equations.
    """

    # Definitions
    ibc_type: str

    def __init__(
        self,
        parameters: Optional[Dict] = None,
        ibc_type: str = "convex-up",
        **kwargs
    ) -> None:
        r"""
        Constructor method
        """
        super().__init__(parameters=parameters, **kwargs)
        logging.info("gme.core.equations_extended.EquationsIdtxIbc")
        self.ibc_type = ibc_type

        self.prep_ibc_eqns()
        self.define_ibc_eqns()
        self.set_ibc_eqns()


class EquationsSetupOnly(EquationsMixedIn, GeodesicMixin, IdtxMixin, IbcMixin):
    r"""
    Generate methods to perform equation definitions but don't act on them.
    """

    # Definitions
    ibc_type: str

    def __init__(self, ibc_type: str = "convex-up", **kwargs) -> None:
        r"""
        Constructor method.
        """
        logging.info("gme.core.equations_extended.EquationsSetup")
        self.ibc_type = ibc_type

        super().__init__(**kwargs)


#

import numpy as np
from gme.core.equations import Equations
from gme.core.equations_extended import EquationsIdtx
from gme.plot.base import Graphing
from gmplib.parameters import Parameters
from sympy import Eq
from typing import Any, Dict, Optional, Tuple, Union

class IndicatrixNew(Graphing):
    H_parametric_eqn: Eq
    tanbeta_max: float
    px_H_lambda: Any
    p_infc_array: np.ndarray
    p_supc_array: np.ndarray
    v_from_gstar_lambda: Any
    v_infc_array: np.ndarray
    v_supc_array: np.ndarray
    def __init__(
        self,
        gmeq: Union[Equations, EquationsIdtx],
        pr: Parameters,
        sub_: Dict,
        varphi_: float = ...,
    ): ...
    def convex_concave_annotations(
        self, do_zoom: bool, eta_: float
    ) -> None: ...
    def Fstar_F_rectlinear(
        self,
        gmeq: Union[Equations, EquationsIdtx],
        job_name: str,
        pr: Parameters,
        do_zoom: bool = ...,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
    ) -> None: ...
    def Fstar_F_polar(
        self,
        gmeq: Union[Equations, EquationsIdtx],
        job_name: str,
        pr: Parameters,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
    ) -> None: ...

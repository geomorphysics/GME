import numpy as np
from gme.core.equations import Equations
from gme.plot.base import Graphing
from sympy import Eq, Symbol
from typing import Any, Dict, List, Optional, Tuple, Union

class SlicingMath:
    H_Ci_eqn: Eq
    degCi_H0p5_eqn: Eq
    gstarhat_eqn: Eq
    pxhat_lambda: Any
    pzhat_lambda: Any
    def __init__(
        self,
        gmeq: Equations,
        sub_: Dict,
        var_list: List[Symbol],
        do_modv: bool = ...,
    ): ...
    H_lambda: Any
    def define_H_lambda(self, sub_: Dict, var_list: List[Symbol]) -> None: ...
    d2Hdpzhat2_lambda: Any
    def define_d2Hdpzhat2_lambda(
        self, sub_: Dict, var_list: List[Symbol]
    ) -> None: ...
    detHessianSqrd_lambda: Any
    def define_detHessianSqrd_lambda(
        self, sub_: Dict, var_list: List[Symbol]
    ) -> None: ...
    Ci_lambda: Any
    def define_Ci_lambda(self, sub_: Dict, var_list: List[Symbol]) -> None: ...
    gstar_signature_lambda: Any
    def define_Hessian_eigenvals(
        self, sub_: Dict, var_list: List[Symbol]
    ) -> None: ...
    gstarhat_lambda: Any
    def define_gstarhat_lambda(
        self, sub_: Dict, var_list: List[Symbol]
    ) -> None: ...
    v_pxpzhat_lambda: Any
    def define_v_pxpzhat_lambda(self, sub_: Dict) -> None: ...
    modv_pxpzhat_lambda: Any
    def define_modv_pxpzhat_lambda(self, sub_: Dict) -> None: ...
    def pxhatsqrd_Ci_polylike_eqn(self, sub_: Dict, pzhat_: float) -> Eq: ...
    def pxhat_Ci_soln(
        self, eqn_: Eq, sub_: Dict, rxhat_: float, tolerance: float = ...
    ) -> float: ...
    def pxpzhat0_values(
        self, contour_values_: Union[List[float], Tuple[float]], sub_: Dict
    ) -> List[Tuple[float, float]]: ...
    def get_rxhat_pzhat(self, sub_: Dict[Any, Any]) -> List[float]: ...

class SlicingPlots(Graphing):
    H_Ci_eqn: Eq
    degCi_H0p5_eqn: Eq
    gstarhat_eqn: Eq
    grid_array: np.ndarray
    pxpzhat_grids: List[np.ndarray]
    rxpxhat_grids: List[np.ndarray]
    def __init__(
        self,
        gmeq: Equations,
        grid_res: int = ...,
        dpi: int = ...,
        font_size: int = ...,
    ) -> None: ...
    def plot_dHdp_slice(
        self,
        sm: SlicingMath,
        sub_: Dict,
        psub_: Dict,
        pxhat_: float,
        do_detHessian: bool = ...,
        do_at_rxcrit: bool = ...,
    ) -> str: ...
    def plot_modv_slice(
        self, sm: SlicingMath, sub_: Dict, psub_: Dict, do_at_rxcrit: bool = ...
    ) -> str: ...
    def H_rxpx_contours(
        self, sm: SlicingMath, sub_: Dict, psf: float, do_Ci: bool, **kwargs
    ) -> str: ...
    def H_pxpz_contours(
        self, sm: SlicingMath, sub_: Dict, psf: float, do_Ci: bool, **kwargs
    ) -> str: ...
    def plot_Hetc_contours(
        self,
        sm: SlicingMath,
        grids_: Tuple[Any, Any],
        sub_: Dict,
        do_Ci: bool,
        do_modv: bool = ...,
        do_fmt_labels: bool = ...,
        do_aspect: bool = ...,
        do_rxpx: bool = ...,
        pxpz_points: Any | None = ...,
        do_log2H: bool = ...,
        do_siggrid: bool = ...,
        do_black_contours: bool = ...,
        do_grid: bool = ...,
        do_at_rxcrit: bool = ...,
        contour_nlevels: Optional[Union[int, List, Tuple]] = ...,
        contour_range: Tuple[float, float] = ...,
        v_contour_range: Tuple[float, float] = ...,
        contour_values: Optional[List[float]] = ...,
        contour_label_locs: Optional[List] = ...,
    ) -> str: ...

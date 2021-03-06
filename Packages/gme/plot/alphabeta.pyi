import numpy as np
from gme.core.equations import Equations
from gme.plot.base import Graphing
from typing import Optional, Tuple

class AlphaBeta(Graphing):
    def alpha_beta(
        self,
        gmeq: Equations,
        name: str,
        alpha_array: np.ndarray,
        beta_array: np.ndarray,
        tanalpha_ext_: float,
        tanbeta_crit_: float,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
    ) -> None: ...
    def beta_anisotropy(
        self,
        gmeq: Equations,
        name: str,
        alpha_array: np.ndarray,
        beta_array: np.ndarray,
        tanalpha_ext_: float,
        tanbeta_crit_: float,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
    ) -> None: ...
    def alpha_anisotropy(
        self,
        gmeq: Equations,
        name: str,
        alpha_array: np.ndarray,
        beta_array: np.ndarray,
        tanalpha_ext_: float,
        tanbeta_crit_: float,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
    ) -> None: ...
    def alpha_image(
        self,
        gmeq: Equations,
        name: str,
        alpha_array: np.ndarray,
        beta_array: np.ndarray,
        tanalpha_ext_: float,
        tanbeta_crit_: float,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
    ) -> None: ...

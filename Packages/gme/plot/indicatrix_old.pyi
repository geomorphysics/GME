import numpy as np
from gme.plot.base import Graphing
from matplotlib.pyplot import Axes
from sympy import Eq
from typing import Callable, Dict, Tuple

class IndicatrixOld(Graphing):
    def comparison_logpolar(
        self,
        gmeq: Eq,
        name: str,
        fig_size: Tuple[float, float] = ...,
        dpi: float = ...,
        varphi_: float = ...,
        n_points: int = ...,
        idtx_pz_min: float = ...,
        idtx_pz_max: float = ...,
        fgtx_pz_min: float = ...,
        fgtx_pz_max: float = ...,
        y_limits: Tuple[float, float] = ...,
    ) -> None: ...
    def text_labels(
        self,
        gmeq: Eq,
        varphi_: float,
        px_: float,
        pz_: float,
        rdotx_: float,
        rdotz_: float,
        zoom_factor: float,
        do_text_labels: bool,
    ) -> None: ...
    def arrows(
        self, px_: float, pz_: float, rdotx_: float, rdotz_: float
    ) -> None: ...
    def lines_and_points(
        self,
        pd: Dict,
        axes: Axes,
        zoomx: np.ndarray,
        do_pz: bool,
        do_shapes: bool,
    ) -> None: ...
    def annotations(
        self, axes: Axes, beta_: float, tanalpha_: float
    ) -> None: ...
    def legend(
        self,
        gmeq: Eq,
        axes: Axes,
        do_legend: bool,
        do_half: bool,
        do_ray_slowness: bool = ...,
    ) -> None: ...
    def figuratrix(
        self,
        gmeq: Eq,
        varphi_: float,
        n_points: int,
        pz_min_: float = ...,
        pz_max_: float = ...,
    ) -> Tuple[np.ndarray, np.ndarray, Eq]: ...
    def indicatrix(
        self,
        gmeq: Eq,
        varphi_: float,
        n_points: int,
        pz_min_: float = ...,
        pz_max_: float = ...,
    ) -> Tuple[np.ndarray, np.ndarray, Eq, Eq]: ...
    def plot_figuratrix(
        self,
        fgtx_px_array: np.ndarray,
        fgtx_pz_array: np.ndarray,
        maybe_recip_fn: Callable,
        do_ray_slowness: bool = ...,
    ) -> None: ...
    def plot_indicatrix(
        self,
        idtx_rdotx_array: np.ndarray,
        idtx_rdotz_array: np.ndarray,
        maybe_recip_fn: Callable,
        do_ray_slowness: bool = ...,
    ) -> None: ...
    def plot_unit_circle(
        self, varphi_: float, do_varphi_circle: bool
    ) -> None: ...
    def relative_geometry(
        self,
        gmeq: Eq,
        name: str,
        fig_size: Tuple[float, float] = ...,
        dpi: float = ...,
        varphi_: float = ...,
        zoom_factor: float = ...,
        do_half: bool = ...,
        do_legend: bool = ...,
        do_text_labels: bool = ...,
        do_arrows: bool = ...,
        do_lines_points: bool = ...,
        do_shapes: bool = ...,
        do_varphi_circle: bool = ...,
        do_pz: bool = ...,
        do_ray_slowness: bool = ...,
        x_max: float = ...,
        n_points: int = ...,
        pz_min_: float = ...,
    ) -> None: ...

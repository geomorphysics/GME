import numpy as np
from gmplib.plot import GraphingBase
from matplotlib.pyplot import Axes
from typing import Dict, List, Optional, Tuple

class Graphing(GraphingBase):
    def mycolors(
        self,
        i: int,
        r: int,
        n: int,
        do_smooth: bool = ...,
        cmap_choice: str = ...,
    ) -> List[str]: ...
    def gray_color(
        self, i_isochrone: int = ..., n_isochrones: int = ...
    ) -> str: ...
    def correct_quadrant(self, angle: float) -> float: ...
    def draw_rays_with_arrows_simple(
        self,
        axes: Axes,
        sub: Dict,
        xi_vh_ratio: float,
        t_array: np.ndarray,
        rx_array: np.ndarray,
        rz_array: np.ndarray,
        v_array: Optional[np.ndarray] = ...,
        n_t: Optional[int] = ...,
        n_rays: int = ...,
        ls: str = ...,
        sf: float = ...,
        color: Optional[str] = ...,
        do_labels: bool = ...,
        do_one_ray: bool = ...,
    ) -> None: ...
    def arrow_annotate_ray_custom(
        self,
        rx_array: np.ndarray,
        rz_array: np.ndarray,
        axes: Axes,
        i_ray: int,
        i_ray_step: int,
        n_rays: int,
        n_arrows: int,
        arrow_sf: float = ...,
        arrow_offset: int = ...,
        x_limits: Optional[Tuple[Optional[float], Optional[float]]] = ...,
        y_limits: Optional[Tuple[Optional[float], Optional[float]]] = ...,
        line_style: str = ...,
        line_width: float = ...,
        ray_label: str = ...,
        do_smooth_colors: bool = ...,
    ) -> None: ...

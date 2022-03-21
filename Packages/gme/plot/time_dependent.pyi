from gme.core.equations import Equations
from gme.ode.velocity_boundary import VelocityBoundarySolution
from gme.plot.base import Graphing
from typing import Dict, Optional, Tuple

class TimeDependent(Graphing):
    def profile_isochrones(
        self,
        gmes: VelocityBoundarySolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        do_zero_isochrone: bool = ...,
        do_rays: bool = ...,
        ray_subsetting: int = ...,
        ray_lw: float = ...,
        ray_ls: str = ...,
        ray_label: str = ...,
        do_isochrones: bool = ...,
        isochrone_subsetting: int = ...,
        isochrone_lw: float = ...,
        isochrone_ls: str = ...,
        do_annotate_rays: bool = ...,
        n_arrows: int = ...,
        arrow_sf: float = ...,
        arrow_offset: int = ...,
        do_annotate_cusps: bool = ...,
        cusp_lw: float = ...,
        do_smooth_colors: bool = ...,
        x_limits: Tuple[float, float] = ...,
        y_limits: Tuple[float, float] = ...,
        aspect: float = ...,
        do_legend: bool = ...,
        do_alt_legend: bool = ...,
        do_grid: bool = ...,
        do_infer_initiation: bool = ...,
        do_pub_label: bool = ...,
        do_etaxi_label: bool = ...,
        pub_label: Optional[str] = ...,
        eta_label_xy: Tuple[float, float] = ...,
        pub_label_xy: Tuple[float, float] = ...,
    ) -> None: ...

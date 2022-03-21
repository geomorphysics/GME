from gme.core.equations import Equations
from gme.ode.single_ray import SingleRaySolution
from gme.plot.base import Graphing
from typing import Dict, Optional, Tuple

class RayVelocities(Graphing):
    def profile_v(
        self,
        gmes: SingleRaySolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        n_points: int = ...,
        do_pub_label: bool = ...,
        pub_label: str = ...,
        pub_label_xy: Tuple[float, float] = ...,
        do_etaxi_label: bool = ...,
        eta_label_xy: Tuple[float, float] = ...,
        var_label_xy: Tuple[float, float] = ...,
        xi_norm: Optional[float] = ...,
        legend_loc: str = ...,
        do_mod_v: bool = ...,
    ) -> None: ...
    def profile_vdot(
        self,
        gmes: SingleRaySolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        n_points: int = ...,
        do_pub_label: bool = ...,
        pub_label: str = ...,
        do_etaxi_label: bool = ...,
        xi_norm: Optional[float] = ...,
        legend_loc: str = ...,
        do_legend: bool = ...,
        do_mod_vdot: bool = ...,
        do_geodesic: bool = ...,
    ) -> None: ...

from gme.core.equations import Equations
from gme.core.equations_extended import (
    EquationsGeodesic,
    EquationsIbc,
    EquationsIdtx,
)
from gme.ode.single_ray import SingleRaySolution
from gme.ode.time_invariant import TimeInvariantSolution
from gme.ode.velocity_boundary import VelocityBoundarySolution
from gme.plot.base import Graphing
from typing import Dict, Optional, Tuple, Union

class RayAngles(Graphing):
    def alpha_beta(
        self,
        gmes: Union[
            SingleRaySolution, TimeInvariantSolution, VelocityBoundarySolution
        ],
        gmeq: Union[Equations, EquationsGeodesic, EquationsIdtx, EquationsIbc],
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        aspect: float = ...,
        n_points: int = ...,
        x_limits: Optional[Tuple[Optional[float], Optional[float]]] = ...,
        y_limits: Optional[Tuple[Optional[float], Optional[float]]] = ...,
        do_legend: bool = ...,
        do_pub_label: bool = ...,
        pub_label_xy: Tuple[float, float] = ...,
        pub_label: str = ...,
        do_etaxi_label: bool = ...,
        eta_label_xy: Tuple[float, float] = ...,
    ) -> None: ...
    def angular_disparity(
        self,
        gmes: Union[
            SingleRaySolution, TimeInvariantSolution, VelocityBoundarySolution
        ],
        gmeq: Union[Equations, EquationsGeodesic, EquationsIdtx, EquationsIbc],
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        n_points: int = ...,
        x_limits: Optional[Tuple[Optional[float], Optional[float]]] = ...,
        y_limits: Optional[Tuple[Optional[float], Optional[float]]] = ...,
        do_legend: bool = ...,
        aspect: float = ...,
        do_pub_label: bool = ...,
        pub_label: str = ...,
        pub_label_xy: Tuple[float, float] = ...,
        eta_label_xy: Tuple[float, float] = ...,
    ) -> None: ...
    def profile_angular_disparity(
        self,
        gmes: Union[
            SingleRaySolution, TimeInvariantSolution, VelocityBoundarySolution
        ],
        gmeq: Union[Equations, EquationsGeodesic, EquationsIdtx, EquationsIbc],
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        n_points: int = ...,
        do_pub_label: bool = ...,
        pub_label: str = ...,
        pub_label_xy: Tuple[float, float] = ...,
        eta_label_xy: Tuple[float, float] = ...,
        var_label_xy: Tuple[float, float] = ...,
    ) -> None: ...
    def profile_alpha(
        self,
        gmes: Union[
            SingleRaySolution, TimeInvariantSolution, VelocityBoundarySolution
        ],
        gmeq: Union[Equations, EquationsGeodesic, EquationsIdtx, EquationsIbc],
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        n_points: int = ...,
        do_legend: bool = ...,
        eta_label_xy: Tuple[float, float] = ...,
    ) -> None: ...
    def psi_eta_alpha(
        self,
        gmeq: Union[Equations, EquationsGeodesic, EquationsIdtx, EquationsIbc],
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        n_points: int = ...,
    ) -> None: ...

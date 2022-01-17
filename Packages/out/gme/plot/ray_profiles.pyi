from gme.core.equations import Equations
from gme.ode.single_ray import SingleRaySolution
from gme.ode.time_invariant import TimeInvariantSolution
from gme.plot.base import Graphing
from typing import Any, Dict, Optional, Tuple

class RayProfiles(Graphing):
    def profile_ray(self, gmes: SingleRaySolution, gmeq: Equations, sub: Dict, name: str, fig_size: Optional[Tuple[float, float]] = ..., dpi: Optional[int] = ..., y_limits: Optional[Tuple[Optional[float], Optional[float]]] = ..., n_points: int = ..., aspect: Optional[float] = ..., do_schematic: bool = ..., do_ndim: bool = ..., do_simple: bool = ..., do_t_sampling: bool = ..., do_pub_label: bool = ..., pub_label: str = ..., pub_label_xy: Tuple[float, float] = ..., do_etaxi_label: bool = ..., eta_label_xy: Any | None = ...) -> None: ...
    def profile_h_rays(self, gmes: TimeInvariantSolution, gmeq: Equations, sub: Dict, name: str, fig_size: Optional[Tuple[float, float]] = ..., dpi: Optional[int] = ..., x_limits: Optional[Tuple[Optional[float], Optional[float]]] = ..., y_limits: Optional[Tuple[Optional[float], Optional[float]]] = ..., n_points: int = ..., do_direct: bool = ..., n_rays: int = ..., do_schematic: bool = ..., do_legend: bool = ..., do_fault_bdry: bool = ..., do_compute_xivh_ratio: bool = ..., do_one_ray: bool = ..., do_t_sampling: bool = ..., do_pub_label: bool = ..., pub_label: str = ..., pub_label_xy: Tuple[float, float] = ..., do_etaxi_label: bool = ..., eta_label_xy: Tuple[float, float] = ...) -> None: ...
    def profile_h(self, gmes: TimeInvariantSolution, gmeq: Equations, sub: Dict, name: str, fig_size: Optional[Tuple[float, float]] = ..., dpi: Optional[int] = ..., y_limits: Optional[Tuple[Optional[float], Optional[float]]] = ..., do_legend: bool = ..., do_profile_points: bool = ..., profile_subsetting: int = ..., do_pub_label: bool = ..., pub_label: str = ..., pub_label_xy: Tuple[float, float] = ..., do_etaxi_label: bool = ..., eta_label_xy: Tuple[float, float] = ...) -> None: ...

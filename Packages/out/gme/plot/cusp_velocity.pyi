from gme.core.equations import Equations
from gme.ode.velocity_boundary import VelocityBoundarySolution
from gme.plot.base import Graphing
from typing import Dict, Optional, Tuple

class CuspVelocity(Graphing):
    def profile_cusp_horizontal_speed(self, gmes: VelocityBoundarySolution, gmeq: Equations, sub: Dict, name: str, fig_size: Optional[Tuple[float, float]] = ..., dpi: Optional[int] = ..., x_limits: Tuple[float, float] = ..., y_limits: Tuple[Optional[float], Optional[float]] = ..., t_limits: Tuple[Optional[float], Optional[float]] = ..., legend_loc: str = ..., do_x: bool = ..., do_infer_initiation: bool = ...) -> None: ...

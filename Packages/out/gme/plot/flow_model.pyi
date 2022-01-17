from gme.core.equations import Equations
from gme.plot.base import Graphing
from typing import Dict, Optional, Tuple

class FlowModel(Graphing):
    def profile_flow_model(self, gmeq: Equations, sub: Dict, name: str, fig_size: Optional[Tuple[float, float]] = ..., dpi: Optional[int] = ..., n_points: int = ..., subtitle: str = ..., do_subtitling: bool = ..., do_extra_annotations: bool = ...) -> None: ...

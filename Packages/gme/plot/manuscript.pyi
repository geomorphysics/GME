from gme.plot.base import Graphing
from typing import Any, Optional, Tuple

class Manuscript(Graphing):
    def point_pairing(
        self,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
    ) -> None: ...
    def covector_isochrones(
        self,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
    ) -> None: ...
    def huygens_wavelets(
        self,
        gmes,
        gmeq,
        sub,
        name,
        fig_size: Any | None = ...,
        dpi: Any | None = ...,
        do_ray_conjugacy: bool = ...,
        do_fast: bool = ...,
        do_legend: bool = ...,
        legend_fontsize: int = ...,
        annotation_fontsize: int = ...,
    ) -> None: ...

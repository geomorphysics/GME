from gme.core.equations import Equations
from gme.ode.time_invariant import TimeInvariantSolution
from gme.plot.base import Graphing
from typing import Dict, Optional, Tuple

class TimeInvariant(Graphing):
    fc: str
    def profile_aniso(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        n_points: int = ...,
        xf_stop: float = ...,
        sf: Optional[Tuple[float, float]] = ...,
        n_arrows: int = ...,
        y_limits: Optional[Tuple[float, float]] = ...,
        v_scale: float = ...,
        v_exponent: float = ...,
        do_pub_label: bool = ...,
        pub_label: str = ...,
        eta_label_xy: Optional[Tuple[float, float]] = ...,
    ) -> None: ...
    def profile_beta(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        n_points: int = ...,
        xf_stop: float = ...,
        legend_loc: str = ...,
        eta_label_xy: Tuple[float, float] = ...,
        pub_label_xy: Tuple[float, float] = ...,
        do_etaxi_label: bool = ...,
        do_pub_label: bool = ...,
        pub_label: str = ...,
    ) -> None: ...
    def profile_beta_error(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        n_points: int = ...,
        eta_label_xy: Tuple[float, float] = ...,
        xf_stop: float = ...,
    ) -> None: ...
    def profile_xi(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        xf_stop: float = ...,
        n_points: int = ...,
        pub_label_xy: Tuple[float, float] = ...,
        eta_label_xy: Tuple[float, float] = ...,
        var_label_xy: Tuple[float, float] = ...,
        do_etaxi_label: bool = ...,
        do_pub_label: bool = ...,
        pub_label: str = ...,
        xi_norm: Optional[float] = ...,
    ) -> None: ...
    def profile_xihorizontal(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        xf_stop: float = ...,
        n_points: int = ...,
        pub_label_xy: Tuple[float, float] = ...,
        eta_label_xy: Tuple[float, float] = ...,
        var_label_xy: Tuple[float, float] = ...,
        do_etaxi_label: bool = ...,
        do_pub_label: bool = ...,
        pub_label: str = ...,
        xi_norm: Optional[float] = ...,
    ) -> None: ...
    def profile_xivertical(
        self,
        gmes: TimeInvariantSolution,
        gmeq: Equations,
        sub: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        xf_stop: float = ...,
        n_points: int = ...,
        y_limits: Optional[Tuple[float, float]] = ...,
        pub_label_xy: Tuple[float, float] = ...,
        eta_label_xy: Tuple[float, float] = ...,
        var_label_xy: Tuple[float, float] = ...,
        do_etaxi_label: bool = ...,
        do_pub_label: bool = ...,
        pub_label: str = ...,
        xi_norm: Optional[float] = ...,
    ) -> None: ...
    def profile_ensemble(
        self,
        gmes: TimeInvariantSolution,
        pr_choices: Dict,
        name: str,
        fig_size: Optional[Tuple[float, float]] = ...,
        dpi: Optional[int] = ...,
        aspect: Optional[float] = ...,
        do_direct: bool = ...,
    ) -> None: ...

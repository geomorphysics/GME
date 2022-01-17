from gme.core.equations import Equations, EquationsMixedIn
from gme.core.geodesic import GeodesicMixin
from gme.core.ibc import IbcMixin
from gme.core.idtx import IdtxMixin
from typing import Any, Dict

class EquationsGeodesic(Equations, GeodesicMixin):
    def __init__(self, parameters: Dict, **kwargs) -> None: ...

class EquationsIdtx(Equations, IdtxMixin):
    def __init__(self, parameters: Dict, **kwargs) -> None: ...

class EquationsIbc(Equations, IbcMixin):
    ibc_type: Any
    def __init__(self, parameters: Dict, ibc_type: str = ..., **kwargs) -> None: ...

class EquationsIdtxIbc(EquationsIdtx, IbcMixin):
    ibc_type: Any
    def __init__(self, parameters: Dict, ibc_type: str = ..., **kwargs) -> None: ...

class EquationsSetupOnly(EquationsMixedIn, GeodesicMixin, IdtxMixin, IbcMixin):
    ibc_type: Any
    def __init__(self, ibc_type: str = ..., **kwargs) -> None: ...

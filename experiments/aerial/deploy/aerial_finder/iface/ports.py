"""External ports. Adapters behind these may use private stacks."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from .schema import AirTaskIn, GroundTaskIn, HealthOut, ParseIn, ParseOut


class ParserPort(Protocol):
    def parse(self, req: ParseIn) -> ParseOut: ...


class GroundPort(Protocol):
    def start(self, req: GroundTaskIn) -> Dict[str, Any]: ...
    def status(self) -> Dict[str, Any]: ...
    def abort(self) -> Dict[str, Any]: ...
    def ready(self) -> str: ...


class AirPort(Protocol):
    def start(self, req: AirTaskIn) -> Dict[str, Any]: ...
    def status(self) -> Dict[str, Any]: ...
    def ready(self) -> str: ...


class LinkPort(Protocol):
    def status(self) -> Dict[str, Any]: ...
    def cmd(self, body: Dict[str, Any]) -> Dict[str, Any]: ...
    def ready(self) -> str: ...


class Facade:
    def __init__(self, parser: ParserPort, ground: Optional[GroundPort] = None,
                 air: Optional[AirPort] = None, link: Optional[LinkPort] = None) -> None:
        self.parser = parser
        self.ground = ground
        self.air = air
        self.link = link

    def health(self) -> HealthOut:
        return HealthOut(
            ok=True,
            ground=self.ground.ready() if self.ground else "offline",
            air=self.air.ready() if self.air else "offline",
            link=self.link.ready() if self.link else "offline",
        )

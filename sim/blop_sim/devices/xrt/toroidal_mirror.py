"""KB mirror devices for XRTBackend."""

from ophyd_async.core import StandardReadable, soft_signal_rw
from ophyd_async.core import StandardReadableFormat as Format

from ...backends import SimBackend


class ToroidalMirror(StandardReadable):
    """Toroidal Mirror with curvature radius control (for XRTBackend).

    Exposes a single radius parameter that directly controls the XRT mirror R value.
    Used with XRTBackend for ray-tracing simulation.

    Args:
        backend: Simulation backend (should be XRTBackend)
        roll: Initial roll of second crystal in mm
        name: Device name
    """

    def __init__(
        self,
        backend: SimBackend,
        yaw: float,
        lateral: float,
        name: str = "",
    ):
        self._backend = backend

        # Curvature radius signal
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.yaw = soft_signal_rw(float, yaw)
            self.lateral = soft_signal_rw(float, lateral)

        super().__init__(name=name)

        # Register with backend
        backend.register_device(
            device_name=name,
            device_type="toroidal_mirror_xrt",
            get_state_callback=self._get_state,
        )

    async def _get_state(self) -> dict:
        """Get current mirror state for backend (async)."""
        return {
            "yaw": await self.yaw.get_value(),
            "lateral": await self.lateral.get_value(),
        }


__all__ = ["ToroidalMirror"]

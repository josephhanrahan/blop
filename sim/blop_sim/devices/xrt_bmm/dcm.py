"""KB mirror devices for XRTBackend."""

from ophyd_async.core import StandardReadable, soft_signal_rw
from ophyd_async.core import StandardReadableFormat as Format

from ...backends import SimBackend


class DCM(StandardReadable):
    """DCM with second crystal roll control (for XRTBMMBackend).

    Exposes the roll of the second crystal of the DCM
    Used with XRTBMMBackend for ray-tracing simulation.

    Args:
        backend: Simulation backend (should be XRTBMMBackend)
        roll: Initial roll of second crystal in mm
        name: Device name
    """

    def __init__(
        self,
        backend: SimBackend,
        roll: float,
        name: str = "",
    ):
        self._backend = backend

        # second crystal roll
        with self.add_children_as_readables(Format.HINTED_SIGNAL):
            self.roll = soft_signal_rw(float, roll)

        super().__init__(name=name)

        # Register with backend
        backend.register_device(
            device_name=name,
            device_type="dcm_xrt",
            get_state_callback=self._get_state,
        )

    async def _get_state(self) -> dict:
        """Get current DCM state for backend (async)."""
        return {
            "roll": await self.roll.get_value(),
        }


__all__ = ["DCM"]

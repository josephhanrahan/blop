"""XRT ray-tracing beam simulation backend."""

import numpy as np

from . import SimBackend
from .models.xrt_bmm_model import build_beamline, build_histRGB, run_process

class XRTBMMBackend(SimBackend):
    """XRT ray-tracing simulation backend.

    Uses the XRT package to perform realistic ray-tracing through a KB mirror pair.
    Much slower than SimpleBackend but more physically accurate.
    """

    def __init__(self, noise: bool = False):
        """Initialize XRT backend."""
        super().__init__()
        self._beamline = None
        # self._limits = [[-0.6, 0.6], [-0.45, 0.45]]
        self._limits = [[-5, 5], [-5, 5]]

        self._noise = noise

    def _ensure_beamline(self):
        """Build XRT beamline if not already built."""
        if self._beamline is None:
            self._beamline = build_beamline()

    def change_energy(self, ev):
        self._beamline = build_beamline(ev)

    async def generate_beam(self) -> np.ndarray:
        """Generate beam using XRT ray-tracing.

        Returns:
            2D numpy array with shape (300, 400)
        """
        self._ensure_beamline()

        # Get KB mirror radii from devices
        dcm_roll = await self._get_dcm_roll()

        # Update XRT beamline mirror parameters
        # self._beamline.toroidMirror01.R = mirror_radii[0]  # Vertical mirror
        # self._beamline.toroidMirror02.R = mirror_radii[1]  # Horizontal mirror
        # self._beamline.dcM01.cryst2roll = dcm_roll

        self._beamline.dcM01.cryst2roll = dcm_roll
        print("ROLL: {}".format(self._beamline.dcM01.cryst2roll))

        # TODO match to correct toroidMirror attrs
        m2_yaw = await self._get_m2_yaw()
        print("YAW await: {}".format(m2_yaw))
        self._beamline.toroidMirror01.yaw = m2_yaw
        print("YAW: {}".format(self._beamline.toroidMirror01.yaw))

        m2_lateral = await self._get_m2_lateral()
        print("LATERAL await: {}".format(m2_lateral))
        self._beamline.toroidMirror01.center[0] = m2_lateral
        print("LATERAL: {}".format(self._beamline.toroidMirror01.center))
        # Run ray tracing
        outDict = run_process(self._beamline)
        lb = outDict["screen04_local"]
        print("LB: ", lb)

        # Build histogram from ray data
        hist2d, _, _ = build_histRGB(lb, lb, limits=self._limits, isScreen=True, shape=[400, 300])
        image = hist2d

        # Add noise if requested
        if self._noise:
            image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))

        return image

    async def _get_dcm_roll(self) -> float:
        """Get KB mirror radii from registered devices.

        Returns:
            [R1, R2] where R1 is first mirror (vertical), R2 is second mirror (horizontal)
        """

        for name, device in self._device_states.items():
            if device["type"] == "dcm_xrt":
                state = await self._get_device_state(name)
                return state['roll']
    
    async def _get_m2_yaw(self) -> float:
        for name, device in self._device_states.items():
            if device["type"] == "toroidal_mirror_xrt":
                state = await self._get_device_state(name)
                return state["yaw"]
    
    async def _get_m2_lateral(self) -> float:
        for name, device in self._device_states.items():
            if device["type"] == "toroidal_mirror_xrt":
                state = await self._get_device_state(name)
                return state['lateral']


__all__ = ["XRTBMMBackend"]

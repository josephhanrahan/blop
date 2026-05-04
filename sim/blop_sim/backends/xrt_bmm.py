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

        # get and set DCM roll
        dcm_roll = await self._get_dcm_roll()
        self._beamline.DCM.cryst2roll = dcm_roll
        print("ROLL: {}".format(self._beamline.DCM.cryst2roll))

        # get and set TFM (m2) yaw
        m2_yaw = await self._get_m2_yaw()
        self._beamline.M2_TFM.yaw = m2_yaw
        print("YAW: {}".format(self._beamline.M2_TFM.yaw))

        # get and set TFM (m2) lateral (x-position)
        m2_lateral = await self._get_m2_lateral()
        self._beamline.M2_TFM.center[0] = m2_lateral
        print("LATERAL: {}".format(self._beamline.M2_TFM.center))

        # Run ray tracing
        outDict = run_process(self._beamline)
        lb = outDict["XAS_SAMPLE_local"]
        # print("LB: ", lb)

        # Build histogram from ray data
        hist2d, _, _ = build_histRGB(lb, lb, limits=self._limits, isScreen=True, shape=[400, 300])

        # hist2d, _, _ = build_histRGB(lb, lb, limits = [[-0.6, 0.6], [-0.45, 0.45]], isScreen=True, shape=[400, 300])
        image = hist2d

        # Add noise if requested (pretty sure XRT already adds noise? don't think this is needed)
        if self._noise:
            image += 1e-3 * np.abs(np.random.standard_normal(size=image.shape))

        return image

    async def _get_dcm_roll(self) -> float:
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

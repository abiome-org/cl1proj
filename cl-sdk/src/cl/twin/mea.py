from __future__ import annotations

import numpy as np


class MEAGeometry:
    """
    Spatial model for the 64-channel CL1-style MEA.

    Channels are represented as an 8x8 grid in a virtual 1000um x 1000um tissue
    plane.  The geometry is intentionally simple, but all twin coupling flows
    through it so future work can replace it with calibrated electrode maps.
    """

    def __init__(self, channel_count: int = 64, tissue_size_um: float = 1000.0):
        if channel_count != 64:
            raise ValueError("The current twin geometry supports the common 64-channel MEA.")
        self.channel_count = channel_count
        self.tissue_size_um = tissue_size_um
        axis = np.linspace(0.0, tissue_size_um, 8)
        xx, yy = np.meshgrid(axis, axis)
        self.channel_xy_um = np.column_stack([xx.ravel(), yy.ravel()])

        deltas = self.channel_xy_um[:, None, :] - self.channel_xy_um[None, :, :]
        self.distance_um = np.linalg.norm(deltas, axis=2)
        # Avoid singularities for self-coupling while preserving strong local effects.
        self.distance_um[self.distance_um == 0.0] = tissue_size_um / 16.0

    def attenuation_from_channel(self, channel: int, gamma: float = 1.25) -> np.ndarray:
        """
        Return distance-decayed coupling from one stimulating electrode to all channels.

        This is the MEA-level analogue of the volume-conductor approximation in
        the north-star design.  The returned vector is normalized to one at the
        strongest affected channel.
        """
        distances = self.distance_um[int(channel)]
        attenuation = 1.0 / np.power(distances, gamma)
        return attenuation / attenuation.max()

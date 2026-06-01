from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mea import MEAGeometry


@dataclass(frozen=True)
class TissueTopology:
    """
    Spatial virtual tissue used by cell-level twin engines.

    The topology places neurons in the same 1000um x 1000um coordinate system
    as the MEA, maps each cell to its nearest recording electrode, and computes
    electrode field attenuation into every cell.  Keeping this as a separate
    artifact lets both Izhikevich and future detailed SNN engines share the same
    biological interface assumptions.
    """

    neuron_xy_um: np.ndarray
    neuron_channel: np.ndarray
    electrode_to_neuron_attenuation: np.ndarray
    neuron_distance_um: np.ndarray

    @classmethod
    def random(
        cls,
        *,
        neuron_count: int,
        mea: MEAGeometry,
        rng: np.random.Generator,
        field_gamma: float = 1.25,
        channel_density: np.ndarray | None = None,
        compute_pairwise_distances: bool = True,
    ) -> "TissueTopology":
        """Create a random 2D virtual culture over the MEA plane."""
        if channel_density is None:
            neuron_xy_um = rng.uniform(0.0, mea.tissue_size_um, size=(neuron_count, 2))
        else:
            neuron_xy_um = cls._sample_from_channel_density(
                neuron_count    = neuron_count,
                mea             = mea,
                rng             = rng,
                channel_density = channel_density,
            )
        electrode_delta = neuron_xy_um[:, None, :] - mea.channel_xy_um[None, :, :]
        electrode_distances = np.linalg.norm(electrode_delta, axis=2)
        neuron_channel = np.argmin(electrode_distances, axis=1).astype(np.int64)

        safe_electrode_distances = np.maximum(electrode_distances, mea.tissue_size_um / 64.0)
        attenuation = 1.0 / np.power(safe_electrode_distances, field_gamma)
        attenuation = attenuation.T
        attenuation /= np.maximum(attenuation.max(axis=1, keepdims=True), 1e-12)

        if compute_pairwise_distances:
            neuron_delta = neuron_xy_um[:, None, :] - neuron_xy_um[None, :, :]
            neuron_distance_um = np.linalg.norm(neuron_delta, axis=2)
        else:
            # Large sparse engines only need positions and electrode fields.
            # Avoiding an N x N distance matrix is the difference between a
            # practical 5k-cell SDK twin and a memory-heavy dense toy model.
            neuron_distance_um = np.zeros((0, 0), dtype=np.float64)

        return cls(
            neuron_xy_um                     = neuron_xy_um,
            neuron_channel                   = neuron_channel,
            electrode_to_neuron_attenuation  = attenuation,
            neuron_distance_um               = neuron_distance_um,
        )

    @staticmethod
    def _sample_from_channel_density(
        *,
        neuron_count: int,
        mea: MEAGeometry,
        rng: np.random.Generator,
        channel_density: np.ndarray,
    ) -> np.ndarray:
        """
        Sample virtual cells around electrodes according to calibrated density.

        The density vector is a coarse prior inferred from recording activity,
        not an exact cell segmentation.  Cells are jittered around the selected
        electrode with a spread near one grid spacing so local populations remain
        spatially continuous instead of collapsing onto electrode centers.
        """
        density = np.asarray(channel_density, dtype=np.float64)
        if density.shape != (mea.channel_count,) or np.sum(density) <= 0.0:
            return rng.uniform(0.0, mea.tissue_size_um, size=(neuron_count, 2))
        density = np.clip(density, 0.0, None)
        density /= np.sum(density)

        channels = rng.choice(mea.channel_count, size=neuron_count, p=density)
        spacing_um = mea.tissue_size_um / 7.0
        jitter = rng.normal(0.0, spacing_um * 0.35, size=(neuron_count, 2))
        neuron_xy_um = mea.channel_xy_um[channels] + jitter
        return np.clip(neuron_xy_um, 0.0, mea.tissue_size_um)

    def distance_decayed_synapses(
        self,
        *,
        rng: np.random.Generator,
        excitatory_mask: np.ndarray,
        connection_probability: float,
        length_constant_um: float,
    ) -> np.ndarray:
        """
        Build directed cell-to-cell weights with distance-decayed probability.

        Excitatory cells produce positive weights and inhibitory cells produce
        negative weights.  The matrix is intentionally dense enough for small
        SDK simulations but bounded; larger future engines can switch to sparse
        storage behind the same method.
        """
        probability = connection_probability * np.exp(
            -self.neuron_distance_um / max(length_constant_um, 1.0)
        )
        np.fill_diagonal(probability, 0.0)
        connected = rng.random(probability.shape) < probability
        weights = rng.uniform(0.02, 0.12, size=probability.shape) * connected
        # Synapses are stored target-by-source, so inhibitory identity belongs
        # on columns: every outgoing synapse from an inhibitory cell is negative.
        weights[:, ~excitatory_mask] *= -1.0
        return weights.astype(np.float64)

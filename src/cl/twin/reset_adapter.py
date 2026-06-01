"""
Compatibility import for the reset SNN CL SDK adapter.

New code should import ``ResetSNNAdapter`` from ``cl1_clsdk_bridge``.
"""

from cl1_clsdk_bridge import ResetSNNAdapter

__all__ = ["ResetSNNAdapter"]

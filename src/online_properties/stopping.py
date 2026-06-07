"""Compatibility wrapper for online_properties.stopping."""

from mlcache.online.stopping import (
    OnlineStoppingConfig,
    OnlineStoppingController,
    WindowedOnlineStoppingController,
)

__all__ = ["OnlineStoppingConfig", "OnlineStoppingController", "WindowedOnlineStoppingController"]


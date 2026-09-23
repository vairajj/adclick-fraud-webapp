"""
V2 inference pipeline for the Ad Click Attribution / Suspicious Click Detection project.

The V2 model expects 20 features:

    ip
    app
    device
    os
    channel
    hour
    day
    day_of_week

    ip_prev_clicks
    app_prev_clicks
    device_prev_clicks
    os_prev_clicks
    channel_prev_clicks

    ip_app_prev_clicks
    ip_device_prev_clicks
    ip_os_prev_clicks
    ip_channel_prev_clicks

    seconds_since_prev_ip_click
    seconds_since_prev_ip_app_click
    seconds_since_prev_ip_device_click

Historical features are calculated BEFORE the current click is added
to the history, matching the chronological V2 feature-engineering logic.
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd


# ============================================================
# Exact feature order used by the V2 model
# ============================================================

V2_FEATURES = [
    "ip",
    "app",
    "device",
    "os",
    "channel",
    "hour",
    "day",
    "day_of_week",
    "ip_prev_clicks",
    "app_prev_clicks",
    "device_prev_clicks",
    "os_prev_clicks",
    "channel_prev_clicks",
    "ip_app_prev_clicks",
    "ip_device_prev_clicks",
    "ip_os_prev_clicks",
    "ip_channel_prev_clicks",
    "seconds_since_prev_ip_click",
    "seconds_since_prev_ip_app_click",
    "seconds_since_prev_ip_device_click",
]


class ClickHistory:
    """
    In-memory store of previously observed clicks.

    This is intentionally session-local for now.

    The structure can later be replaced with SQLite, PostgreSQL,
    Redis, etc. without changing the model itself.
    """

    def __init__(self):
        # Counts of previous events
        self.ip_counts = defaultdict(int)
        self.app_counts = defaultdict(int)
        self.device_counts = defaultdict(int)
        self.os_counts = defaultdict(int)
        self.channel_counts = defaultdict(int)

        # Counts for combinations
        self.ip_app_counts = defaultdict(int)
        self.ip_device_counts = defaultdict(int)
        self.ip_os_counts = defaultdict(int)
        self.ip_channel_counts = defaultdict(int)

        # Last timestamps
        self.last_ip_time: Dict[int, pd.Timestamp] = {}
        self.last_ip_app_time: Dict[Tuple[int, int], pd.Timestamp] = {}
        self.last_ip_device_time: Dict[Tuple[int, int], pd.Timestamp] = {}

        self.total_clicks = 0

    @staticmethod
    def _seconds_since(
        current_time: pd.Timestamp,
        previous_time: pd.Timestamp | None,
    ) -> float:
        """
        Return seconds since previous event.

        V2 uses -1 for cases where there is no previous event.
        """
        if previous_time is None:
            return -1.0

        return float((current_time - previous_time).total_seconds())

    def build_features(
        self,
        ip: int,
        app: int,
        device: int,
        os_value: int,
        channel: int,
        click_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Build V2 features for the current click.

        IMPORTANT:
        Historical counts/timestamps are read BEFORE this click
        is added to the history.
        """

        ip_app_key = (ip, app)
        ip_device_key = (ip, device)
        ip_os_key = (ip, os_value)
        ip_channel_key = (ip, channel)

        row = {
            # Original fields
            "ip": ip,
            "app": app,
            "device": device,
            "os": os_value,
            "channel": channel,

            # Time features
            "hour": click_time.hour,
            "day": click_time.day,
            "day_of_week": click_time.dayofweek,

            # Previous counts
            "ip_prev_clicks": self.ip_counts[ip],
            "app_prev_clicks": self.app_counts[app],
            "device_prev_clicks": self.device_counts[device],
            "os_prev_clicks": self.os_counts[os_value],
            "channel_prev_clicks": self.channel_counts[channel],

            # Previous interaction counts
            "ip_app_prev_clicks": self.ip_app_counts[ip_app_key],
            "ip_device_prev_clicks": self.ip_device_counts[ip_device_key],
            "ip_os_prev_clicks": self.ip_os_counts[ip_os_key],
            "ip_channel_prev_clicks": self.ip_channel_counts[ip_channel_key],

            # Previous click gaps
            "seconds_since_prev_ip_click": self._seconds_since(
                click_time,
                self.last_ip_time.get(ip),
            ),

            "seconds_since_prev_ip_app_click": self._seconds_since(
                click_time,
                self.last_ip_app_time.get(ip_app_key),
            ),

            "seconds_since_prev_ip_device_click": self._seconds_since(
                click_time,
                self.last_ip_device_time.get(ip_device_key),
            ),
        }

        return pd.DataFrame([row], columns=V2_FEATURES)

    def record_click(
        self,
        ip: int,
        app: int,
        device: int,
        os_value: int,
        channel: int,
        click_time: pd.Timestamp,
    ) -> None:
        """
        Add the current click to the history AFTER prediction.
        """

        ip_app_key = (ip, app)
        ip_device_key = (ip, device)
        ip_os_key = (ip, os_value)
        ip_channel_key = (ip, channel)

        # Update counts
        self.ip_counts[ip] += 1
        self.app_counts[app] += 1
        self.device_counts[device] += 1
        self.os_counts[os_value] += 1
        self.channel_counts[channel] += 1

        # Update interaction counts
        self.ip_app_counts[ip_app_key] += 1
        self.ip_device_counts[ip_device_key] += 1
        self.ip_os_counts[ip_os_key] += 1
        self.ip_channel_counts[ip_channel_key] += 1

        # Update timestamps
        self.last_ip_time[ip] = click_time
        self.last_ip_app_time[ip_app_key] = click_time
        self.last_ip_device_time[ip_device_key] = click_time

        self.total_clicks += 1

    def clear(self) -> None:
        """Clear all stored click history."""

        self.__init__()

    def stats(self) -> dict:
        """Return basic history statistics."""

        return {
            "total_clicks": self.total_clicks,
            "unique_ips": len(self.ip_counts),
            "unique_apps": len(self.app_counts),
            "unique_devices": len(self.device_counts),
            "unique_os": len(self.os_counts),
            "unique_channels": len(self.channel_counts),
        }


def validate_click_values(
    ip: int,
    app: int,
    device: int,
    os_value: int,
    channel: int,
) -> None:
    """
    Basic validation for categorical IDs.
    """

    values = {
        "ip": ip,
        "app": app,
        "device": device,
        "os": os_value,
        "channel": channel,
    }

    for name, value in values.items():
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer.")

        if value < 0:
            raise ValueError(f"{name} cannot be negative.")


def build_v2_features(
    history: ClickHistory,
    ip: int,
    app: int,
    device: int,
    os_value: int,
    channel: int,
    click_time: datetime,
) -> pd.DataFrame:
    """
    Public helper used by the API.
    """

    validate_click_values(
        ip=ip,
        app=app,
        device=device,
        os_value=os_value,
        channel=channel,
    )

    timestamp = pd.Timestamp(click_time)

    return history.build_features(
        ip=ip,
        app=app,
        device=device,
        os_value=os_value,
        channel=channel,
        click_time=timestamp,
    )
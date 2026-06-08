"""Implementation of the spatiotemporal constraint evidence agent.

This agent estimates a developer's habitual UTC offset range from
timestamped activity logs.  According to the AgentLocD paper, the
spatiotemporal agent provides a physical consistency constraint that
prunes candidate locations produced by the semantic and collaboration
agents rather than independently producing a complete location vector.

The implementation uses simple heuristics:

* Fetch all event timestamps for a developer from the ``opensource.events`` table.
* Compute a distribution of activity by hour of day in UTC.
* Estimate the most likely local time offset by aligning the activity
  distribution to typical working hours (assumed to be around 9:00–17:00).
* Return a range around the estimated offset defined by a fixed
  tolerance ``epsilon``.

The returned range is expressed as a tuple ``(utc_offset_min, utc_offset_max)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import logging
import pandas as pd
from datetime import datetime, timezone
import numpy as np

from .db_utils import fetch_dataframe
from .llm_client import BailianQwenClient
from .prompts import TEMPORAL_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class TimeConstraint:
    """Represents a UTC offset range derived from temporal evidence."""

    offset_min: float
    offset_max: float
    evidence: str


class SpatiotemporalConstraintAgent:
    """Estimate UTC offset ranges from timestamped activity logs."""

    def __init__(self, epsilon_hours: float = 1.0, llm_client: Optional[BailianQwenClient] = None, use_llm: bool = True) -> None:
        """Create a new SpatiotemporalConstraintAgent.

        Parameters
        ----------
        epsilon_hours : float, default 1.0
            The tolerance around the estimated UTC offset in hours.  The
            final range will be ``(offset - epsilon, offset + epsilon)``.
        """
        self.epsilon = epsilon_hours
        self.llm_client = llm_client
        self.use_llm = use_llm


    def infer_llm(self, actor_login: str) -> Optional[Dict[str, Any]]:
        """Return an LLM-interpreted temporal constraint.

        The UTC-offset range is calculated deterministically, then passed to
        A_time's role-specific prompt so the model can produce warnings and a
        structured temporal rationale.
        """
        constraint = self.infer(actor_login)
        if constraint is None:
            return None
        payload = {
            "actor_login": actor_login,
            "utc_offset_range": [constraint.offset_min, constraint.offset_max],
            "deterministic_evidence": constraint.evidence,
        }
        if not self.use_llm or self.llm_client is None:
            return {
                "agent": "A_time",
                "utc_offset_range": [constraint.offset_min, constraint.offset_max],
                "supported_longitudes": [],
                "warnings": [],
                "rationale": constraint.evidence,
            }
        return self.llm_client.chat_json(TEMPORAL_PROMPT, payload)

    def infer(self, actor_login: str) -> Optional[TimeConstraint]:
        """Estimate the UTC offset range for a developer.

        Parameters
        ----------
        actor_login : str
            The developer login used to query event timestamps.

        Returns
        -------
        TimeConstraint or None
            The estimated UTC offset range, or None if no events are available.
        """
        # Fetch timestamps from events table.  We only need created_at.
        query = """
        SELECT created_at
        FROM opensource.events
        WHERE actor_login = %(actor)s
        ORDER BY created_at
        """
        df = fetch_dataframe(query, params={"actor": actor_login})
        if df.empty:
            return None

        # Convert timestamps to datetime and extract hour of day (UTC).
        timestamps = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        timestamps = timestamps.dropna()
        if timestamps.empty:
            return None

        hours = timestamps.dt.hour
        # Compute histogram of activity hours
        counts = hours.value_counts().sort_index()
        # Reindex to full 24 hours so that missing hours have zero count
        full_index = pd.Index(range(24))
        counts = counts.reindex(full_index, fill_value=0)

        # Convert counts to numpy array for computation
        counts_array = counts.values.astype(float)
        # Compute circular mean of activity distribution (0–23 hours)
        # Convert hours to radians: hour * 2π / 24
        angles = 2 * np.pi * np.arange(24) / 24
        weighted_cos = np.sum(counts_array * np.cos(angles))
        weighted_sin = np.sum(counts_array * np.sin(angles))
        mean_angle = np.arctan2(weighted_sin, weighted_cos)
        # Convert mean angle back to hour (0–24)
        mean_hour = (mean_angle % (2 * np.pi)) * 24 / (2 * np.pi)

        # Assume typical activity peaks during local working hours (9–17).
        # Estimate UTC offset by aligning mean_hour to noon (12:00).
        utc_offset_est = (mean_hour - 12.0) / 1.0  # hours
        # Normalise offset to within [-12, +12]
        if utc_offset_est > 12:
            utc_offset_est -= 24
        elif utc_offset_est < -12:
            utc_offset_est += 24

        offset_min = utc_offset_est - self.epsilon
        offset_max = utc_offset_est + self.epsilon
        return TimeConstraint(offset_min=offset_min, offset_max=offset_max,
                              evidence=f"mean_hour:{mean_hour:.2f}")
"""Top‑level pipeline for running the AgentLocD geolocation inference.

This module ties together the three evidence agents and the
orchestrator.  It defines a convenience class ``AgentLocDPipeline``
that can be used to process a list of developers and produce
geolocation predictions.  The predictions may be written to a
CSV file for downstream analysis or integrated into other systems.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .profile_semantic_agent import ProfileSemanticAgent
from .ecosystem_collaboration_agent import EcosystemCollaborationAgent
from .spatiotemporal_constraint_agent import SpatiotemporalConstraintAgent
from .orchestrator import Orchestrator


class AgentLocDPipeline:
    """Convenience wrapper for running the AgentLocD inference pipeline."""

    def __init__(
        self,
        institution_domain_map: Optional[Dict[str, str]] = None,
        max_collaboration_anchors: int = 10,
        temporal_epsilon: float = 1.0,
    ) -> None:
        """Initialise the pipeline with agent configuration.

        Parameters
        ----------
        institution_domain_map : dict, optional
            Mapping from email domain to institution location used by
            the profile‑semantic agent.
        max_collaboration_anchors : int, default 10
            Maximum number of collaboration anchors used by the
            collaboration agent.
        temporal_epsilon : float, default 1.0
            The tolerance in hours used by the spatiotemporal agent.
        """
        self.semantic_agent = ProfileSemanticAgent(institution_domain_map=institution_domain_map)
        self.collaboration_agent = EcosystemCollaborationAgent(max_anchors=max_collaboration_anchors)
        self.temporal_agent = SpatiotemporalConstraintAgent(epsilon_hours=temporal_epsilon)
        self.orchestrator = Orchestrator(
            semantic_agent=self.semantic_agent,
            collaboration_agent=self.collaboration_agent,
            temporal_agent=self.temporal_agent,
        )

    def infer_developer(self, actor_login: str, name: Optional[str] = None,
                         email: Optional[str] = None, text: Optional[str] = None,
                         org: Optional[str] = None) -> Tuple[Optional[str], Dict[str, float], Dict[str, str]]:
        """Infer the best location for a single developer.

        See :meth:`Orchestrator.infer_location` for details on the return values.
        """
        return self.orchestrator.infer_location(
            actor_login=actor_login,
            name=name,
            email=email,
            text=text,
            org=org,
        )

    def run(self, developers: Iterable[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]]) -> pd.DataFrame:
        """Run geolocation inference for a list of developers.

        Parameters
        ----------
        developers : iterable of tuples
            An iterable of (actor_login, name, email, text, org) tuples.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns ``login``, ``predicted_location``,
            ``scores`` and ``evidence``.  ``scores`` and ``evidence``
            columns contain dictionaries mapping candidate locations to
            aggregated scores and evidence traces, respectively.
        """
        results = []
        for login, name, email, text, org in developers:
            pred, scores, evidence = self.infer_developer(login, name=name, email=email, text=text, org=org)
            results.append({
                "login": login,
                "predicted_location": pred,
                "scores": scores,
                "evidence": evidence,
            })
        return pd.DataFrame(results)
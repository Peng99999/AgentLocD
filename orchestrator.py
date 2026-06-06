"""Implementation of the AgentLocD orchestrator.

The orchestrator coordinates the three evidence agents, combines
candidate locations, and applies the Free-MAD arbitration mechanism
with graceful degradation.  In the full AgentLocD paper the
Free‑MAD algorithm performs multi‑round debate among agents to
resolve conflicts at each spatial tier.  For the purposes of this
reference implementation we provide a simplified arbitration
procedure that still respects the reliability ordering of agents and
demonstrates how the agents interact.

The orchestrator proceeds as follows:

1. Invoke the profile‑semantic, collaboration and temporal agents for
   a given developer login to obtain candidate location sets and a
   UTC offset range.
2. Consolidate candidates by location string, summing scores where
   both agents agree.
3. Apply a reliability prior: collaboration candidates are preferred
   over semantic candidates, and temporal constraints may prune
   candidates whose UTC offset appears inconsistent (not implemented
   in this simplified version because location strings are not tied
   to offsets).
4. Select the highest scoring candidate.  If no candidates are
   available, return ``None`` indicating insufficient evidence.
5. Return the selected candidate and the collected evidence traces.

Note that this simplified orchestrator does not infer hierarchical
location tiers; it merely produces a single best guess location
string.  Developers wishing to replicate the full AgentLocD
functionality should extend this class to handle multi‑level spatial
hierarchies, implement a proper debate mechanism, and incorporate
temporal pruning based on UTC offsets.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Dict

import logging

from .profile_semantic_agent import ProfileSemanticAgent, Candidate as SemanticCandidate
from .ecosystem_collaboration_agent import EcosystemCollaborationAgent, CollaborationCandidate
from .spatiotemporal_constraint_agent import SpatiotemporalConstraintAgent, TimeConstraint

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinate evidence agents and produce final geolocation predictions."""

    def __init__(
        self,
        semantic_agent: ProfileSemanticAgent,
        collaboration_agent: EcosystemCollaborationAgent,
        temporal_agent: SpatiotemporalConstraintAgent,
    ) -> None:
        self.semantic_agent = semantic_agent
        self.collaboration_agent = collaboration_agent
        self.temporal_agent = temporal_agent

    def infer_location(self, actor_login: str, name: Optional[str] = None,
                       email: Optional[str] = None, text: Optional[str] = None,
                       org: Optional[str] = None) -> Tuple[Optional[str], Dict[str, float], Dict[str, str]]:
        """Infer a developer's location using all agents.

        Parameters
        ----------
        actor_login : str
            Developer login for collaboration and temporal evidence lookup.
        name, email, text, org : optional
            Profile fields used by the semantic agent.

        Returns
        -------
        Tuple[Optional[str], Dict[str, float], Dict[str, str]]
            A tuple containing the selected location (or None), a
            mapping from candidate locations to aggregated scores,
            and a mapping from candidate locations to concatenated
            evidence strings.
        """
        # Collect evidence from each agent
        sem_candidates = self.semantic_agent.infer(actor_login, name=name, email=email, text=text, org=org)
        collab_candidates = self.collaboration_agent.infer(actor_login)
        time_constraint = self.temporal_agent.infer(actor_login)

        # Consolidate candidates
        scores: Dict[str, float] = {}
        evidence: Dict[str, str] = {}

        # Add collaboration candidates first (higher priority)
        for cand in collab_candidates:
            loc = cand.location
            scores[loc] = scores.get(loc, 0.0) + cand.score
            evidence[loc] = evidence.get(loc, '') + f"collab({cand.evidence})"

        # Add semantic candidates (lower priority) – only if not present or to boost existing
        for cand in sem_candidates:
            loc = cand.location
            # If collab has already produced this location we'll still add the semantic score
            scores[loc] = scores.get(loc, 0.0) + cand.score
            ev_prev = evidence.get(loc, '')
            sep = ';' if ev_prev else ''
            evidence[loc] = ev_prev + sep + f"sem({cand.evidence})"

        # Apply temporal constraint pruning (not fully implemented):
        # In a complete implementation we would map each location to a
        # time zone and remove candidates outside ``time_constraint``.
        # Here we simply log the constraint for debugging.
        if time_constraint:
            logger.debug(
                "Time constraint for %s: offset range %.2f to %.2f (evidence=%s)",
                actor_login, time_constraint.offset_min, time_constraint.offset_max, time_constraint.evidence
            )

        if not scores:
            return None, scores, evidence

        # Select the highest scoring location
        best_loc = max(scores, key=scores.get)
        return best_loc, scores, evidence
"""Implementation of the ecosystem collaboration evidence agent.

This agent infers geographic candidates from a developer's collaboration
behaviour in the open source ecosystem.  The underlying assumption is
not that collaborators are always geographically close, but rather
that sustained participation in regionally anchored projects,
organisations or language communities can provide complementary
evidence when profile fields are sparse.

The implementation uses OpenDigger's OpenRank metrics to identify
repositories and organisations that best represent a developer's
collaboration footprint.  Two levels of OpenRank are available:

* **Global OpenRank** – the overall influence of a developer in the
  global open source community across all collaborations.  This is
  accessed via the ``opensource.global_openrank`` table.

* **Community OpenRank** – the influence of a developer within a
  particular repository or organisation.  This is accessed via the
  ``opensource.community_openrank`` table and provides a finer‑grained
  view of a developer's interactions.

For each developer, the agent selects a core collaboration set of
repositories and organisations by sorting community OpenRank scores.
It then aggregates geographic labels associated with those anchors.
In this simplified implementation we treat the organisation login and
repository name as the candidate labels when no external mapping to
location is available.  In a production system these anchors should be
resolved to actual geographic locations via a separate knowledge base
or mapping service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import logging

import pandas as pd

from .db_utils import fetch_dataframe

logger = logging.getLogger(__name__)


@dataclass
class CollaborationCandidate:
    """Represents a candidate geographic label derived from collaboration evidence."""

    location: str
    score: float
    evidence: str


class EcosystemCollaborationAgent:
    """Compute candidate locations based on collaboration evidence and OpenRank."""

    def __init__(self, max_anchors: int = 10) -> None:
        """Create a new EcosystemCollaborationAgent.

        Parameters
        ----------
        max_anchors : int, default 10
            The maximum number of repositories or organisations to
            consider when building the core collaboration set.  Higher
            values yield more candidates but increase query cost.
        """
        self.max_anchors = max_anchors

    def infer(self, actor_login: str) -> List[CollaborationCandidate]:
        """Generate candidate locations for a developer based on collaboration evidence.

        Parameters
        ----------
        actor_login : str
            The developer login used to query collaboration data.

        Returns
        -------
        list[CollaborationCandidate]
            A list of candidate locations with associated scores.
        """
        # Fetch community openrank records for this actor.  We select
        # repositories and organisations where the actor is present and
        # order by OpenRank descending.  We limit the number of rows to
        # ``max_anchors`` to avoid overloading the query engine.
        community_query = f"""
        SELECT
            org_login,
            repo_name,
            openrank
        FROM opensource.community_openrank
        WHERE actor_login = %(actor)s
        ORDER BY openrank DESC
        LIMIT %(limit)s
        """
        community_df = fetch_dataframe(community_query, params={"actor": actor_login, "limit": self.max_anchors})

        candidates: Dict[str, CollaborationCandidate] = {}

        # Aggregate scores by organisation login and repository name.
        # In the absence of an external mapping from org/repo to actual
        # geographic location, we treat the org login and repo name as
        # labels.  Users of the framework may replace this logic with a
        # lookup into a knowledge base that maps projects/organisations
        # to cities or countries.
        for _, row in community_df.iterrows():
            org_login = row["org_login"]
            repo_name = row["repo_name"]
            openrank = row["openrank"] or 0.0

            if pd.notna(org_login) and org_login:
                loc = org_login
                self._add_candidate(candidates, loc, openrank, f"org_openrank:{org_login}")

            if pd.notna(repo_name) and repo_name:
                loc = repo_name
                self._add_candidate(candidates, loc, openrank * 0.5, f"repo_openrank:{repo_name}")

        # Return candidates sorted by score descending
        return sorted(candidates.values(), key=lambda c: c.score, reverse=True)

    def _add_candidate(self, candidates: Dict[str, CollaborationCandidate],
                       location: str, score: float, evidence: str) -> None:
        """Helper to add or update a collaboration candidate."""
        if location in candidates:
            cand = candidates[location]
            cand.score += score
            cand.evidence += f";{evidence}"
        else:
            candidates[location] = CollaborationCandidate(location=location, score=score, evidence=evidence)
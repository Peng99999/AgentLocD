"""AgentLocD orchestrator with Free-MAD style LLM arbitration.

The orchestrator receives structured candidate sets from A_sem, A_org and
A_time, places them on a shared blackboard, and calls Qwen with an
orchestrator prompt to perform debate-style arbitration. It then applies
lightweight graceful degradation as a post-check to avoid unsupported lower-tier
completion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging

from .ecosystem_collaboration_agent import EcosystemCollaborationAgent
from .llm_client import BailianQwenClient
from .llm_types import FourTierCandidate
from .profile_semantic_agent import ProfileSemanticAgent
from .prompts import ORCHESTRATOR_PROMPT
from .spatiotemporal_constraint_agent import SpatiotemporalConstraintAgent

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinate evidence agents and produce final four-tier predictions."""

    def __init__(
        self,
        semantic_agent: ProfileSemanticAgent,
        collaboration_agent: EcosystemCollaborationAgent,
        temporal_agent: SpatiotemporalConstraintAgent,
        llm_client: Optional[BailianQwenClient] = None,
        max_debate_rounds: int = 3,
        use_llm: bool = True,
    ) -> None:
        self.semantic_agent = semantic_agent
        self.collaboration_agent = collaboration_agent
        self.temporal_agent = temporal_agent
        self.llm_client = llm_client
        self.max_debate_rounds = max_debate_rounds
        self.use_llm = use_llm

    def infer_location(
        self,
        actor_login: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        text: Optional[str] = None,
        org: Optional[str] = None,
        location: Optional[str] = None,
        cleaned_location_mentions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Infer a four-tier spatial vector for one developer."""
        sem_candidates = self.semantic_agent.infer_llm(
            actor_login,
            name=name,
            email=email,
            text=text,
            org=org,
            location=location,
            cleaned_location_mentions=cleaned_location_mentions,
        )
        org_candidates = self.collaboration_agent.infer_llm(actor_login)
        time_constraint = self.temporal_agent.infer_llm(actor_login)

        blackboard = {
            "developer": {"login": actor_login},
            "max_debate_rounds": self.max_debate_rounds,
            "profile_semantic_candidates": [c.to_dict() for c in sem_candidates],
            "collaboration_candidates": [c.to_dict() for c in org_candidates],
            "temporal_constraint": time_constraint,
        }

        if self.use_llm and self.llm_client is not None:
            decision = self.llm_client.chat_json(ORCHESTRATOR_PROMPT, blackboard)
        else:
            decision = self._fallback_arbitration(sem_candidates, org_candidates, time_constraint)

        decision = self._apply_graceful_degradation(decision, sem_candidates, org_candidates)
        decision["blackboard"] = blackboard
        return decision

    def _fallback_arbitration(
        self,
        sem_candidates: List[FourTierCandidate],
        org_candidates: List[FourTierCandidate],
        time_constraint: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Deterministic fallback when no LLM client is configured."""
        candidates = sorted(
            org_candidates + sem_candidates,
            key=lambda c: (c.confidence, 1 if c.agent == "A_org" else 0),
            reverse=True,
        )
        if not candidates:
            return {
                "country": None,
                "state": None,
                "city": None,
                "locality": None,
                "confidence": 0.0,
                "decision_trace": ["No candidate returned by A_sem or A_org."],
                "debate_rounds": [],
            }
        best = candidates[0]
        return {
            "country": best.country,
            "state": best.state,
            "city": best.city,
            "locality": best.locality,
            "confidence": best.confidence,
            "decision_trace": [f"Fallback selected top candidate from {best.agent}: {best.rationale}"],
            "debate_rounds": [],
        }

    def _apply_graceful_degradation(
        self,
        decision: Dict[str, Any],
        sem_candidates: List[FourTierCandidate],
        org_candidates: List[FourTierCandidate],
    ) -> Dict[str, Any]:
        """Post-check lower-tier outputs against semantic/collaboration evidence.

        This is intentionally conservative: if a city or locality value appears
        without any supporting semantic/collaboration candidate at that tier, it
        is truncated to None along with lower tiers.
        """
        support = [c for c in sem_candidates + org_candidates]

        def supported(tier: str, value: Optional[str]) -> bool:
            if value is None:
                return True
            return any(getattr(c, tier) == value for c in support)

        if not supported("country", decision.get("country")):
            decision.update({"country": None, "state": None, "city": None, "locality": None})
            decision.setdefault("decision_trace", []).append("Graceful degradation removed unsupported country tier.")
        elif not supported("state", decision.get("state")):
            decision.update({"state": None, "city": None, "locality": None})
            decision.setdefault("decision_trace", []).append("Graceful degradation removed unsupported state and lower tiers.")
        elif not supported("city", decision.get("city")):
            decision.update({"city": None, "locality": None})
            decision.setdefault("decision_trace", []).append("Graceful degradation removed unsupported city and locality tiers.")
        elif not supported("locality", decision.get("locality")):
            decision.update({"locality": None})
            decision.setdefault("decision_trace", []).append("Graceful degradation removed unsupported locality tier.")
        return decision

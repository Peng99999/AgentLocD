"""Top-level pipeline for running AgentLocD with LLM-backed agents."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .ecosystem_collaboration_agent import EcosystemCollaborationAgent
from .llm_client import BailianQwenClient, LLMConfig
from .orchestrator import Orchestrator
from .profile_semantic_agent import ProfileSemanticAgent
from .spatiotemporal_constraint_agent import SpatiotemporalConstraintAgent


class AgentLocDPipeline:
    """Convenience wrapper for running the AgentLocD inference pipeline."""

    def __init__(
        self,
        institution_domain_map: Optional[Dict[str, str]] = None,
        max_collaboration_anchors: int = 10,
        temporal_epsilon: float = 1.0,
        use_llm: bool = True,
        llm_config: Optional[LLMConfig] = None,
    ) -> None:
        self.use_llm = use_llm
        self.llm_client = BailianQwenClient(config=llm_config) if use_llm else None
        self.semantic_agent = ProfileSemanticAgent(
            institution_domain_map=institution_domain_map,
            llm_client=self.llm_client,
            use_llm=use_llm,
        )
        self.collaboration_agent = EcosystemCollaborationAgent(
            max_anchors=max_collaboration_anchors,
            llm_client=self.llm_client,
            use_llm=use_llm,
        )
        self.temporal_agent = SpatiotemporalConstraintAgent(
            epsilon_hours=temporal_epsilon,
            llm_client=self.llm_client,
            use_llm=use_llm,
        )
        self.orchestrator = Orchestrator(
            semantic_agent=self.semantic_agent,
            collaboration_agent=self.collaboration_agent,
            temporal_agent=self.temporal_agent,
            llm_client=self.llm_client,
            use_llm=use_llm,
        )

    def infer_developer(
        self,
        actor_login: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        text: Optional[str] = None,
        org: Optional[str] = None,
        location: Optional[str] = None,
        cleaned_location_mentions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self.orchestrator.infer_location(
            actor_login=actor_login,
            name=name,
            email=email,
            text=text,
            org=org,
            location=location,
            cleaned_location_mentions=cleaned_location_mentions,
        )

    def run(
        self,
        developers: Iterable[Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]],
    ) -> pd.DataFrame:
        results: List[Dict[str, Any]] = []
        for login, name, email, text, org in developers:
            pred = self.infer_developer(login, name=name, email=email, text=text, org=org)
            results.append({
                "login": login,
                "country": pred.get("country"),
                "state": pred.get("state"),
                "city": pred.get("city"),
                "locality": pred.get("locality"),
                "confidence": pred.get("confidence"),
                "decision_trace": pred.get("decision_trace"),
                "blackboard": pred.get("blackboard"),
            })
        return pd.DataFrame(results)

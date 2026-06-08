"""Shared dataclasses for LLM-backed AgentLocD components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FourTierCandidate:
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    locality: Optional[str] = None
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    rationale: str = ""
    agent: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any], agent: str) -> "FourTierCandidate":
        return cls(
            country=data.get("country"),
            state=data.get("state"),
            city=data.get("city"),
            locality=data.get("locality"),
            confidence=float(data.get("confidence") or 0.0),
            evidence=list(data.get("evidence") or []),
            rationale=str(data.get("rationale") or ""),
            agent=agent,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "country": self.country,
            "state": self.state,
            "city": self.city,
            "locality": self.locality,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "rationale": self.rationale,
            "agent": self.agent,
        }

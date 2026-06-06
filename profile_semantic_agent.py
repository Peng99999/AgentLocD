"""Implementation of the profile‑semantic evidence agent.

This agent extracts weak and medium strength geographic evidence from
developer profile information.  According to the AgentLocD paper, the
agent should analyse fields such as display names, email domains,
profile texts and organisation metadata to propose candidate
locations.  Because profile information can be missing, humorous,
outdated or intentionally ambiguous the agent does not make a
deterministic prediction.  Instead it constructs a ranked list of
candidate locations with associated confidence scores that will be
considered by the orchestrator alongside other evidence sources.

The implementation below is a practical approximation of the
description in the paper.  It provides several simple heuristics
based on available public data:

* **Name heuristics** – a small optional mapping of first names to
  common countries.  If available, the country frequency is used to
  boost candidate scores.  Users may extend the provided mapping in
  real deployments.

* **Email domain heuristics** – email addresses with institutional
  domains may indicate a developer's affiliation.  The top‑level
  domain is interpreted as a country code when appropriate (e.g.
  ``.fr`` → France).  A separate institution–domain mapping may be
  loaded via the constructor.

* **Textual location extraction** – simple keyword matching against a
  list of country and city names.  Real implementations may replace
  this with a more sophisticated NLP/NER pipeline.

The agent exposes a single ``infer`` method that accepts a developer
record and returns a list of candidates with confidence scores.  The
developer record is expected to contain ``login``, ``name``,
``email``, ``text`` and ``org`` fields, but callers may omit fields if
they are unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import re
import logging

logger = logging.getLogger(__name__)

# A minimal mapping from first names to likely countries.  In a real
# deployment this should be replaced with a comprehensive dataset.
FIRST_NAME_COUNTRY_MAP: Dict[str, str] = {
    # English names
    "john": "United States",
    "mary": "United States",
    "david": "United Kingdom",
    "li": "China",
    "wei": "China",
    # Spanish names
    "carlos": "Spain",
    "maria": "Spain",
    # French names
    "jean": "France",
    # Japanese names
    "taro": "Japan",
    "yuki": "Japan",
}

# A list of country names used for simple keyword matching in profile
# texts.  This list can be extended.  To save space the list includes
# only a representative subset; full implementations should include all
# ISO country names and popular cities.
COUNTRY_KEYWORDS: List[str] = [
    "United States",
    "United Kingdom",
    "China",
    "France",
    "Germany",
    "Japan",
    "Canada",
    "India",
    "Spain",
    "Brazil",
]


@dataclass
class Candidate:
    """Represents a candidate geographic label with an associated score."""

    location: str
    score: float
    evidence: str


class ProfileSemanticAgent:
    """Extract candidate locations from developer profile information."""

    def __init__(
        self,
        institution_domain_map: Optional[Dict[str, str]] = None,
    ) -> None:
        """Create a new ProfileSemanticAgent.

        Parameters
        ----------
        institution_domain_map : dict, optional
            A mapping from email domains to institution or organisation
            names.  If provided this mapping will be used to infer
            candidate locations from institutional domains.  When
            omitted the agent will only use the top‑level domain
            heuristic.
        """
        self.institution_domain_map = institution_domain_map or {}

    def infer(self, login: str, name: Optional[str] = None,
              email: Optional[str] = None, text: Optional[str] = None,
              org: Optional[str] = None) -> List[Candidate]:
        """Generate a ranked list of candidate locations.

        Parameters
        ----------
        login : str
            The developer's username or login.  Used for logging and
            may be passed downstream.
        name : str, optional
            The developer's display name.  Will be tokenised to extract
            potential first names.
        email : str, optional
            The developer's email address.  Used to infer location
            based on domain and institution mapping.
        text : str, optional
            Free‑form profile text or biography.  Used for keyword
            matching of country names.
        org : str, optional
            The developer's organisation affiliation, if any.

        Returns
        -------
        list[Candidate]
            A list of candidate locations with associated scores.  The
            list is sorted in descending order of score.
        """
        candidates: Dict[str, Candidate] = {}

        # 1. Use first name heuristics
        if name:
            tokens = re.split(r"\W+", name.lower())
            for token in tokens:
                country = FIRST_NAME_COUNTRY_MAP.get(token)
                if country:
                    self._add_candidate(candidates, country, 0.2, f"name:{token}")

        # 2. Use email domain heuristics
        if email and "@" in email:
            domain = email.split("@", 1)[1].lower()
            # Check if domain matches an institution mapping
            if domain in self.institution_domain_map:
                inst_loc = self.institution_domain_map[domain]
                self._add_candidate(candidates, inst_loc, 0.4, f"inst_domain:{domain}")
            # Check top‑level domain (e.g. .fr -> France) if not matched above
            parts = domain.split('.')
            if len(parts) > 1:
                tld = parts[-1]
                # naive mapping from country code top level domains to country names
                tld_country_map = {
                    'us': 'United States',
                    'uk': 'United Kingdom',
                    'cn': 'China',
                    'fr': 'France',
                    'de': 'Germany',
                    'jp': 'Japan',
                    'ca': 'Canada',
                    'in': 'India',
                    'es': 'Spain',
                    'br': 'Brazil',
                }
                country_tld = tld_country_map.get(tld)
                if country_tld:
                    self._add_candidate(candidates, country_tld, 0.1, f"tld:{tld}")

        # 3. Use organisation field heuristics
        if org:
            # Very simple heuristic: if organisation string contains a country name
            for country in COUNTRY_KEYWORDS:
                if country.lower() in org.lower():
                    self._add_candidate(candidates, country, 0.3, f"org:{org}")

        # 4. Use profile text for keyword matching
        if text:
            text_lower = text.lower()
            for country in COUNTRY_KEYWORDS:
                if country.lower() in text_lower:
                    self._add_candidate(candidates, country, 0.15, f"text:{country}")

        # Sort candidates by score descending
        return sorted(candidates.values(), key=lambda c: c.score, reverse=True)

    def _add_candidate(self, candidates: Dict[str, Candidate], location: str,
                       score: float, evidence: str) -> None:
        """Helper to update candidate scores with additive weighting.

        If a candidate location is already present, the score is
        incremented.  Evidence strings are concatenated for traceability.

        Parameters
        ----------
        candidates : dict
            The mutable mapping of location strings to Candidate objects.
        location : str
            The geographic label to add or update.
        score : float
            The incremental score to add for this evidence.
        evidence : str
            A string describing the evidence type and value.
        """
        if location in candidates:
            cand = candidates[location]
            cand.score += score
            cand.evidence += f";{evidence}"
        else:
            candidates[location] = Candidate(location=location, score=score, evidence=evidence)
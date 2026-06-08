"""Example script for running the AgentLocD pipeline.

This script demonstrates how to instantiate the pipeline, fetch a
sample of developer logins from the OpenDigger database, run the
geolocation inference on those developers, and save the results to
CSV.  It can be executed directly with `python -m agentlocd.run_pipeline`.
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

import pandas as pd

from .db_utils import fetch_dataframe
from .pipeline import AgentLocDPipeline


def fetch_sample_developers(limit: int = 100) -> List[tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]]:
    """Fetch a sample of developer logins from the events table.

    This helper queries the ``opensource.events`` table for distinct
    ``actor_login`` values and returns a list of tuples containing
    login and empty profile fields (since only the login is used by
    collaboration and temporal agents).  In a real application you
    might join this with a user profile table to obtain names, emails
    and biographies.

    Parameters
    ----------
    limit : int
        Number of developers to fetch.

    Returns
    -------
    list of tuples
        Each tuple is (actor_login, name, email, text, org).  Only
        ``actor_login`` is populated here; other fields are None.
    """
    query = """
    SELECT DISTINCT actor_login
    FROM opensource.events
    WHERE actor_login != ''
    LIMIT %(limit)s
    """
    df = fetch_dataframe(query, params={"limit": limit})
    developers: List[tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]] = []
    for login in df["actor_login"]:
        developers.append((login, None, None, None, None))
    return developers


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AgentLocD pipeline on a sample of developers.")
    parser.add_argument("--limit", type=int, default=50, help="Number of developers to process")
    parser.add_argument("--output", type=str, default="agentlocd_predictions.csv", help="CSV output filename")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM calls and use deterministic fallback")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    # Instantiate pipeline with default settings
    pipeline = AgentLocDPipeline(use_llm=not args.no_llm)
    # Fetch sample developers
    developers = fetch_sample_developers(limit=args.limit)
    logging.info("Fetched %d developers", len(developers))
    # Run inference
    df = pipeline.run(developers)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    logging.info("Saved predictions to %s", args.output)


if __name__ == "__main__":
    main()
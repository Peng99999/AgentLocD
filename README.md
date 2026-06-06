# AgentLocD: Multi-Agent Geolocation Framework

This repository contains the full implementation of **AgentLocD**, a multi‑agent framework for inferring developer locations from heterogeneous open‑source data.  The code here accompanies our ICDM paper on multi‑granularity developer geolocation.  AgentLocD integrates profile‑semantic cues, collaboration relationships, and temporal activity patterns to produce geographic predictions at four administrative levels: country/region, state/province, city/district, and locality.

## Repository contents

- **agentlocd/** – Python package implementing the framework.
  - `db_utils.py`: wraps ClickHouse queries via `db_config.get_clickhouse_client` so you can connect to the OpenDigger database.
  - `profile_semantic_agent.py`: extracts location candidates from names, email domains, profile text and other profile‑semantic signals.
  - `ecosystem_collaboration_agent.py`: leverages OpenRank influence metrics to identify a developer’s core projects and organisations, then weights associated locations by their influence.
  - `spatiotemporal_constraint_agent.py`: computes habitual UTC‑offset ranges from commit and event timestamps to derive temporal constraints.
  - `orchestrator.py`: runs the Free‑MAD debate mechanism, applies a reliability prior to reconcile candidate sets, and invokes graceful degradation to drop unsupported tiers.
  - `pipeline.py`: glues together the agents and orchestrator to perform end‑to‑end inference for a batch of developers.
- **run_pipeline.py** – Example script demonstrating how to load developer IDs from the database, run AgentLocD, and save predictions.
- **location_info_complete.csv.gz** – Public subset of our geolocation dataset.

## About the released dataset

The file `location_info_complete.csv.gz` is a gzipped CSV containing **no developer identifiers or personal data**.  It provides a clean, anonymised subset of our four‑tier location predictions.  Each row represents a location and includes only the following columns:

| Column                          | Meaning                                                            |
|---------------------------------|--------------------------------------------------------------------|
| `location`                      | Standardised location string (e.g., full address or place name)    |
| `country`                       | Country or region (e.g., `United States`, `India`)                 |
| `administrative_area_level_1`   | First‑level subdivision (state, province, or equivalent)           |
| `administrative_area_level_2`   | Second‑level subdivision (county, district, etc.)                  |
| `locality`                      | City, town, or finer locality                                      |

These records have been manually verified to ensure there are **no `NULL` values** and that each row contains a complete four‑tier location.  Because all developer fields have been removed, this dataset can be shared freely and used to benchmark multi‑granularity geolocation methods.  To load the data with pandas:

```python
import pandas as pd
df = pd.read_csv('location_info_complete.csv.gz')

"""AgentLocD package.

This package implements the core components of the AgentLocD framework, a
multi‑agent system for inferring the geographic locations of open‑source
developers across multiple spatial tiers.  The implementation draws from
the methodology described in the associated research paper and is
structured to allow reproducible data extraction and location inference.

Components
==========

The package is organised into several modules:

* ``db_utils``: a small helper module for connecting to the ClickHouse
  database via the provided ``get_clickhouse_client`` function.  All
  database interactions in the agents should use this helper to
  encapsulate the connection logic.
* ``profile_semantic_agent``: implements the
  ``ProfileSemanticAgent``, which extracts candidate locations from
  developer profile information (names, domains, textual fields) and
  cleansed location mentions.
* ``ecosystem_collaboration_agent``: implements the
  ``EcosystemCollaborationAgent``, which uses community and global
  OpenRank scores from collaboration networks to derive location
  candidates.
* ``spatiotemporal_constraint_agent``: implements the
  ``SpatiotemporalConstraintAgent``, which computes habitual UTC offset
  ranges from developers' activity timestamps.
* ``orchestrator``: coordinates the agents via the Free‑MAD debate
  mechanism, resolves conflicting evidence according to a reliability
  prior, and applies graceful degradation when sufficient evidence is
  unavailable.
* ``pipeline``: defines a high‑level function that ties together all
  agents and the orchestrator.  This can be used to infer locations for
  a cohort of developers or to experiment with the system in an end‑to‑
  end fashion.

Usage
-----

To run the pipeline, ensure that the ``db_config`` module is available
in your Python path and provides a ``get_clickhouse_client`` function
that returns a configured ClickHouse client.  Then, you can execute
the ``run_pipeline.py`` script at the root of this repository to
produce location predictions for a sample of developers:

.. code-block:: bash

    python run_pipeline.py --limit 100

This will query the database for the first 100 developers in
``opensource.global_openrank`` and run all agents and the orchestrator
to produce a set of four‑tier location vectors.

Note
----
This reference implementation is intended as a blueprint for
researchers and practitioners.  Many functions are simplified or use
placeholder logic (e.g. geocoding of names/domains) because access to
external services or proprietary data may not be available in all
environments.  Replace these stubs with production‑grade components as
appropriate for your deployment.
"""

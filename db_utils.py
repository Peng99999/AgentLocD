"""Utility functions for connecting to the ClickHouse database used by
AgentLocD.

This module provides a thin wrapper around the ``db_config.get_clickhouse_client``
function made available in the user's environment.  The intent is to
centralise any logic for constructing the ClickHouse client and to
provide convenience helpers for common query patterns.

The returned client is an instance of a clickhouse-driver or
clickhouse-connect client; see the documentation for ``db_config`` for
details.  All queries in the codebase should go through these helpers
to facilitate future instrumentation or caching layers.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import pandas as pd

# Import the database configuration from the provided environment.  The
# ``db_config`` module is expected to define a ``get_clickhouse_client``
# function which returns a connected client object.  The client must
# support a ``query`` method returning an object with a ``result_rows``
# attribute and a ``query_df`` method returning a pandas DataFrame.
try:
    from db_config import get_clickhouse_client  # type: ignore
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "db_config.get_clickhouse_client could not be imported. "
        "Ensure that db_config.py is available in the working directory."
    ) from exc


def get_client() -> Any:
    """Return a new ClickHouse client instance.

    This function is a simple wrapper around ``get_clickhouse_client`` to
    allow a single import location throughout the codebase.  If future
    changes require e.g. configuration injection or connection pooling
    this function can be updated centrally.

    Returns
    -------
    Any
        A connected ClickHouse client object.
    """
    return get_clickhouse_client()


def execute_query(query: str, params: Optional[Iterable[Any]] = None) -> list[tuple[Any, ...]]:
    """Execute a SQL query and return the raw result rows.

    Parameters
    ----------
    query : str
        The SQL query to execute.  Placeholders for parameters should
        conform to the syntax expected by the underlying client.
    params : Iterable[Any], optional
        Optional parameters to pass with the query.

    Returns
    -------
    list[tuple[Any, ...]]
        A list of tuples representing each row returned by the query.
    """
    client = get_client()
    if params is None:
        result = client.query(query)
    else:
        result = client.query(query, params=params)
    return result.result_rows


def fetch_dataframe(query: str, params: Optional[Iterable[Any]] = None) -> pd.DataFrame:
    """Execute a SQL query and return the result as a DataFrame.

    Parameters
    ----------
    query : str
        The SQL query to execute.
    params : Iterable[Any], optional
        Optional parameters to pass with the query.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the query results.
    """
    client = get_client()
    if params is None:
        df = client.query_df(query)
    else:
        df = client.query_df(query, params=params)
    return df
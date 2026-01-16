# btc_forecast/db/utils.py
from __future__ import annotations

import os
from typing import List, Optional

import psycopg2


def _db_env(
    *,
    host_key: str = "DBHOST",
    user_key: str = "DBUSER",
    password_key: str = "DBPASSWORD",
    name_key: str = "DBNAME",
) -> tuple[str, str, str, str]:
    host = os.environ.get(host_key)
    user = os.environ.get(user_key)
    password = os.environ.get(password_key)
    dbname = os.environ.get(name_key)

    missing = [k for k, v in [(host_key, host), (user_key, user), (password_key, password), (name_key, dbname)] if not v]
    if missing:
        raise RuntimeError(f"Missing DB env vars: {missing}")

    return host, user, password, dbname


def get_tracked_coins_from_db(*, limit: Optional[int] = None) -> List[str]:
    host, user, password, dbname = _db_env()

    sql = """
        SELECT symbol
        FROM coin_catalog
        WHERE tracked = TRUE
        ORDER BY symbol
    """
    if limit is not None:
        sql += " LIMIT %s"

    with psycopg2.connect(host=host, user=user, password=password, dbname=dbname) as conn:
        with conn.cursor() as cur:
            if limit is None:
                cur.execute(sql)
            else:
                cur.execute(sql, (int(limit),))
            return [r[0] for r in cur.fetchall()]

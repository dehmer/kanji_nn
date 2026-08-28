import os
from psycopg.conninfo import make_conninfo


def conninfo_from_env(dbname) -> str:
    kwargs = {
        "host": os.getenv("PGHOST", "localhost"),
        "port": os.getenv("PGPORT", "5432"),
        "dbname": os.getenv("PGDATABASE", dbname)
    }

    if user := os.getenv("PGUSER"):
        kwargs["user"] = user

    if password := os.getenv("PGPASSWORD"):
        kwargs["password"] = password

    return make_conninfo(**kwargs)

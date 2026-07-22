import os

from psycopg.conninfo import make_conninfo


def conninfo_from_env() -> str:
    """
    Create a psycopg conninfo string from PostgreSQL environment variables.

    Environment variables:
        PGHOST       (default: localhost)
        PGPORT       (default: 5432)
        PGDATABASE   (required)
        PGUSER       (optional)
        PGPASSWORD   (optional)

    Returns:
        A PostgreSQL conninfo string.

    Raises:
        RuntimeError: If PGDATABASE is not defined.
    """
    dbname = os.getenv("PGDATABASE")
    if not dbname:
        raise RuntimeError("PGDATABASE environment variable is required")

    kwargs = {
        "host": os.getenv("PGHOST", "localhost"),
        "port": os.getenv("PGPORT", "5432"),
        "dbname": dbname,
    }

    if user := os.getenv("PGUSER"):
        kwargs["user"] = user

    if password := os.getenv("PGPASSWORD"):
        kwargs["password"] = password

    return make_conninfo(**kwargs)

"""Alembic environment for laya-mcp.

Two entry points:
- `Repo.migrate()` (app startup, tests): passes its own open connection in
  `config.attributes["connection"]`; migrations run inside that connection's transaction.
- The alembic CLI from the project root (`alembic upgrade head`, `alembic revision --autogenerate`):
  uses `sqlalchemy.url` from alembic.ini if set, else Settings().db_url (LAYA_MCP_DATA_DIR).

render_as_batch=True so ALTERs work on SQLite (table copy-and-move).
"""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlmodel import SQLModel

from laya_mcp.db import models  # noqa: F401  (registers the tables on SQLModel.metadata)

config = context.config

# Only configure logging for CLI runs; programmatic runs keep the application's logging.
if config.config_file_name is not None and "connection" not in config.attributes:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = SQLModel.metadata


def _url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    from laya_mcp.config import get_settings

    return get_settings().db_url


def _configure(**kw) -> None:
    context.configure(target_metadata=target_metadata, render_as_batch=True, compare_type=True, **kw)


def run_migrations_offline() -> None:
    _configure(url=_url(), literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
        return

    from laya_mcp.db.repo import create_sqlite_engine

    engine = create_sqlite_engine(_url())
    try:
        with engine.connect() as conn:
            conn.connection.driver_connection.execute("PRAGMA foreign_keys=OFF")
            with conn.begin():
                _configure(connection=conn)
                with context.begin_transaction():
                    context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

"""Database schema and migration management."""

from mlb.db.schema.migrate import migrate, schema_version

__all__ = ["migrate", "schema_version"]

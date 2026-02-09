"""Forward-only migration runner and schema version helper."""

import asyncio
import re
from pathlib import Path
from typing import Optional

import asyncpg

from mlb.db.pool import get_pool


async def _ensure_migrations_table(conn: asyncpg.Connection) -> None:
    """Ensure schema_migrations table exists."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            filename TEXT NOT NULL DEFAULT '',
            applied_at TIMESTAMPTZ DEFAULT now()
        )
    """)


async def _get_applied_versions(conn: asyncpg.Connection) -> set[int]:
    """Get set of already-applied migration versions."""
    rows = await conn.fetch("SELECT version FROM schema_migrations ORDER BY version")
    return {row["version"] for row in rows}


async def _get_pending_migrations(migrations_dir: Path, applied: set[int]) -> list[tuple[int, Path]]:
    """
    Get list of pending migrations to apply.

    Returns list of (version, path) tuples sorted by version.
    """
    pending = []

    for sql_file in sorted(migrations_dir.glob("*.sql")):
        # Extract version from filename (e.g., "001_initial.sql" -> 1)
        try:
            version = int(sql_file.stem.split("_")[0])
        except (ValueError, IndexError):
            continue

        if version not in applied:
            pending.append((version, sql_file))

    return sorted(pending, key=lambda x: x[0])


def _split_sql_statements(sql: str) -> list[str]:
    """
    Split SQL script into individual statements.

    Handles dollar-quoted strings ($$), single-quoted strings, and comments.
    Uses a state machine to properly track when we're inside quoted sections.
    """
    # Remove single-line comments first (-- ...)
    sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    # Remove multi-line comments (/* ... */)
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)

    statements = []
    current_statement = []
    in_dollar_quote = False
    in_single_quote = False
    i = 0

    while i < len(sql):
        char = sql[i]

        # Check for dollar-quote ($$)
        if sql[i:i+2] == '$$' and not in_single_quote:
            in_dollar_quote = not in_dollar_quote
            current_statement.append('$$')
            i += 2
            continue

        # Check for single quote (')
        if char == "'" and not in_dollar_quote:
            in_single_quote = not in_single_quote
            current_statement.append(char)
            i += 1
            continue

        # Check for semicolon (statement terminator)
        if char == ';' and not in_dollar_quote and not in_single_quote:
            current_statement.append(char)
            stmt = ''.join(current_statement).strip()
            if stmt:
                statements.append(stmt)
            current_statement = []
            i += 1
            continue

        # Regular character
        current_statement.append(char)
        i += 1

    # Add any remaining statement
    if current_statement:
        stmt = ''.join(current_statement).strip()
        if stmt and not stmt.isspace():
            statements.append(stmt)

    return statements


async def _apply_migration(conn: asyncpg.Connection, version: int, sql_path: Path) -> None:
    """Apply a single migration file."""
    sql = sql_path.read_text(encoding="utf-8")
    filename = sql_path.name

    # Split SQL into individual statements and execute each one
    statements = _split_sql_statements(sql)

    for statement in statements:
        if statement.strip():
            await conn.execute(statement)

    # Record migration as applied
    await conn.execute(
        "INSERT INTO schema_migrations (version, filename) VALUES ($1, $2)",
        version,
        filename
    )


async def migrate() -> int:
    """
    Apply all pending migrations in order.

    Uses advisory lock to prevent concurrent migration runs.
    Skips already-applied versions. Idempotent.

    Returns:
        int: Number of migrations applied in this run

    Raises:
        FileNotFoundError: If migrations directory not found
        asyncpg.PostgresError: On database errors
    """
    pool = await get_pool()

    # Find migrations directory using package-relative path (works from any cwd)
    # This resolves to the actual file path even when run via python -m
    migrations_dir = Path(__file__).resolve().parent / "migrations"

    if not migrations_dir.exists() or not migrations_dir.is_dir():
        raise FileNotFoundError(
            f"Migrations directory not found: {migrations_dir}\n"
            f"Expected to find *.sql files in this directory."
        )

    applied_count = 0
    applied_versions = []

    async with pool.acquire() as conn:
        # Acquire advisory lock to prevent concurrent migrations
        # Use lock ID 123456 (arbitrary unique integer)
        lock_acquired = await conn.fetchval("SELECT pg_try_advisory_lock(123456)")

        if not lock_acquired:
            raise RuntimeError(
                "Another migration is currently running. "
                "Wait for it to complete and try again."
            )

        try:
            # Ensure migrations table exists
            await _ensure_migrations_table(conn)

            # Get applied versions
            applied = await _get_applied_versions(conn)

            # Get pending migrations
            pending = await _get_pending_migrations(migrations_dir, applied)

            if not pending:
                return 0

            # Apply each pending migration in a transaction
            for version, sql_path in pending:
                async with conn.transaction():
                    await _apply_migration(conn, version, sql_path)
                    applied_count += 1
                    applied_versions.append((version, sql_path.name))
                    print(f"  [OK] Applied migration {version:03d}: {sql_path.name}")

            return applied_count

        finally:
            # Release advisory lock
            await conn.execute("SELECT pg_advisory_unlock(123456)")


async def schema_version() -> Optional[int]:
    """
    Get the highest applied migration version.

    Returns:
        int: Highest applied version number, or None if no migrations applied

    Raises:
        asyncpg.PostgresError: On database errors
    """
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Ensure migrations table exists
        await _ensure_migrations_table(conn)

        # Get max version
        result = await conn.fetchval(
            "SELECT MAX(version) FROM schema_migrations"
        )

        return result


def main() -> None:
    """CLI entry point for running migrations."""
    async def _run():
        print("Running database migrations...")
        applied = await migrate()
        version = await schema_version()

        if applied == 0:
            print(f"\nNo pending migrations. Current schema version: {version}")
        else:
            print(f"\nSuccessfully applied {applied} migration(s). Current schema version: {version}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()

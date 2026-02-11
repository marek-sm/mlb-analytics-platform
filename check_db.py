import asyncio
import os
import asyncpg

async def main():
    dsn = os.environ["DB_DSN"]
    conn = await asyncpg.connect(dsn)
    row = await conn.fetchrow(
        "SELECT inet_server_addr() AS addr, "
        "inet_server_port() AS port, "
        "current_database() AS db, "
        "current_schema() AS schema"
    )
    print(dict(row))
    await conn.close()

asyncio.run(main())

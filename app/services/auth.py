from __future__ import annotations

import hashlib
import secrets

import aiosqlite
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


async def initialize_db() -> None:
    """Create the api_keys table if it does not exist.

    Called once at application startup before the server begins accepting
    requests. Idempotent — safe to call multiple times.
    """
    async with aiosqlite.connect(settings.database_url.removeprefix("sqlite:///")) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id          INTEGER PRIMARY KEY,
                key_hash    TEXT      UNIQUE NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP,
                is_active   BOOLEAN   DEFAULT TRUE
            )
            """
        )
        await db.commit()
    logger.info("database.initialized", table="api_keys")


def hash_api_key(key: str) -> str:
    """Return the salted SHA-256 hex digest of an API key.

    Only the digest is stored. The salt lives in config, not the DB, so a
    database-only breach can't mount a rainbow-table attack.
    """
    salted = f"{settings.api_key_salt}:{key}"
    return hashlib.sha256(salted.encode()).hexdigest()


async def verify_api_key(key: str) -> bool:
    """Return True if the key exists and is active; touch last_used_at.

    Never raises — returns False on any error so a DB failure degrades to 401,
    not 500. Only the hash prefix is logged, never the plaintext key.
    """
    key_hash = hash_api_key(key)
    log = logger.bind(key_hash=key_hash[:12] + "…")  # log prefix only

    try:
        db_path = settings.database_url.removeprefix("sqlite:///")
        async with aiosqlite.connect(db_path) as db:
            async with db.execute(
                "SELECT id FROM api_keys WHERE key_hash = ? AND is_active = TRUE",
                (key_hash,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                log.warning("auth.key_not_found")
                return False

            await db.execute(
                "UPDATE api_keys SET last_used_at = CURRENT_TIMESTAMP WHERE key_hash = ?",
                (key_hash,),
            )
            await db.commit()

        log.info("auth.key_verified")
        return True

    except Exception:
        log.exception("auth.verify_error")
        return False


async def create_api_key() -> str:
    """Generate, store, and return a new plaintext API key.

    256 bits from the OS CSPRNG. Only the hash is stored; the plaintext is
    returned exactly once and is unrecoverable afterward (like a GitHub PAT).
    """
    plaintext = secrets.token_urlsafe(32)
    key_hash = hash_api_key(plaintext)

    db_path = settings.database_url.removeprefix("sqlite:///")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO api_keys (key_hash) VALUES (?)",
            (key_hash,),
        )
        await db.commit()

    logger.info("auth.key_created", key_hash=key_hash[:12] + "…")
    return plaintext


async def revoke_api_key(key: str) -> bool:
    """Set is_active = FALSE for the given key.

    Returns True if the key existed and was revoked, False if not found.
    Revocation is soft-delete: the row remains for audit history.
    """
    key_hash = hash_api_key(key)
    log = logger.bind(key_hash=key_hash[:12] + "…")

    db_path = settings.database_url.removeprefix("sqlite:///")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "UPDATE api_keys SET is_active = FALSE WHERE key_hash = ? AND is_active = TRUE",
            (key_hash,),
        )
        await db.commit()
        revoked = cursor.rowcount > 0

    if revoked:
        log.info("auth.key_revoked")
    else:
        log.warning("auth.key_not_found_for_revocation")

    return revoked

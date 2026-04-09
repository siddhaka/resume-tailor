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

    We never store plaintext API keys for the same reason we never store
    plaintext passwords: if the database is exfiltrated, the attacker learns
    nothing useful. A raw SHA-256 without a salt would still be vulnerable to
    rainbow-table attacks (precomputed hash→value dictionaries). Including
    api_key_salt — a secret value unique to this deployment — means the
    attacker would need both the database dump *and* the salt to mount any
    offline attack. The salt is stored in Vault / environment variables, not
    in the database, so a database-only breach is insufficient.
    """
    salted = f"{settings.api_key_salt}:{key}"
    return hashlib.sha256(salted.encode()).hexdigest()


async def verify_api_key(key: str) -> bool:
    """Return True if the key exists in the database and is active.

    On success, updates last_used_at so we have an audit trail of key usage.
    Never raises — returns False and logs on any error so that an unexpected
    database failure degrades gracefully to a 401 rather than a 500.

    We log the key_hash rather than the plaintext key so that log aggregators
    (Loki, CloudWatch, etc.) never capture material that could be used to
    impersonate a client.
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

    Uses secrets.token_urlsafe(32) which draws from the OS CSPRNG — the same
    entropy source used for TLS keys. The result is 43 URL-safe base64
    characters (~256 bits of entropy), making brute-force infeasible.

    Only the salted SHA-256 hash is written to the database. The plaintext is
    returned exactly once — this is the user's only opportunity to copy it.
    After this function returns the plaintext is gone forever: there is no
    "show key again" endpoint. This mirrors how GitHub personal access tokens
    and AWS secret keys work, and it is the correct design because it means a
    database breach exposes nothing recoverable.
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

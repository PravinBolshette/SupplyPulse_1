"""
Security, Password Hashing, and JWT Authentication Module.

Implements bcrypt hashing, PyJWT token lifecycle, role-based authorization dependencies,
and PostgreSQL user persistence with resilient local cache fallback.
"""

from datetime import datetime, timedelta, timezone
import os
from typing import Annotated, Any, Dict, List, Optional
from uuid import UUID, uuid4

import bcrypt
import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.auth import UserRole

logger = get_logger("core.security")

# HTTP Bearer security scheme
security_bearer = HTTPBearer(auto_error=False)

# Configuration defaults
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-supplypulse-at-vit-pune-2026-secure")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# In-memory User Store Cache (mirrors PostgreSQL users table)
_USERS_CACHE: Dict[str, Dict[str, Any]] = {}


def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt with standard salt rounds."""
    # Truncate to 72 bytes as per bcrypt specification
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against a bcrypt hash."""
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception as exc:
        logger.warning(f"Password verification failure: {exc}")
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Encodes a JWT payload with expiration and subject claims."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": now})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """Decodes and validates a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _init_default_users() -> None:
    """Seeds default accounts for testing and initial platform access."""
    if _USERS_CACHE:
        return

    defaults = [
        ("analyst", "analyst123", UserRole.ANALYST.value),
        ("manager", "manager123", UserRole.MANAGER.value),
        ("admin", "admin123", UserRole.ADMIN.value),
    ]
    for uname, pwd, role in defaults:
        uid = str(uuid4())
        _USERS_CACHE[uname.lower()] = {
            "user_id": uid,
            "username": uname.lower(),
            "password_hash": hash_password(pwd),
            "role": role,
            "created_at": datetime.now(timezone.utc),
        }


_init_default_users()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieves user from PostgreSQL if connected, or in-memory cache."""
    clean_uname = username.strip().lower()

    # 1. Attempt PostgreSQL retrieval
    db_uri = os.getenv("DATABASE_URL")
    if db_uri and "postgresql" in db_uri:
        try:
            import psycopg2
            conn = psycopg2.connect(db_uri, connect_timeout=1)
            cur = conn.cursor()
            cur.execute(
                "SELECT user_id, username, password_hash, role, created_at FROM users WHERE username = %s;",
                (clean_uname,),
            )
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                user_dict = {
                    "user_id": str(row[0]),
                    "username": row[1],
                    "password_hash": row[2],
                    "role": row[3],
                    "created_at": row[4],
                }
                _USERS_CACHE[clean_uname] = user_dict
                return user_dict
        except Exception:
            pass  # Fall back to cache

    return _USERS_CACHE.get(clean_uname)


def save_user(username: str, password_hash: str, role: str) -> Dict[str, Any]:
    """Persists a new user in PostgreSQL and local cache."""
    clean_uname = username.strip().lower()
    uid = str(uuid4())
    now = datetime.now(timezone.utc)
    user_dict = {
        "user_id": uid,
        "username": clean_uname,
        "password_hash": password_hash,
        "role": role,
        "created_at": now,
    }

    # Persist to PostgreSQL if available
    db_uri = os.getenv("DATABASE_URL")
    if db_uri and "postgresql" in db_uri:
        try:
            import psycopg2
            conn = psycopg2.connect(db_uri, connect_timeout=1)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (user_id, username, password_hash, role, created_at) "
                "VALUES (%s, %s, %s, %s, %s);",
                (uid, clean_uname, password_hash, role, now),
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as exc:
            logger.warning(f"PostgreSQL user insert warning (caching locally): {exc}")

    _USERS_CACHE[clean_uname] = user_dict
    return user_dict


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
) -> Dict[str, Any]:
    """
    FastAPI dependency: Validates the Bearer JWT token from the Authorization header,
    decodes claims, and resolves the authenticated user profile.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided. Bearer token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is missing subject claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_username(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"User '{username}' no longer exists in authentication repository.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def require_roles(allowed_roles: List[str]):
    """
    Role-Based Access Control (RBAC) dependency factory.
    Verifies that the authenticated operator possesses an authorized operational role.
    """
    async def role_checker(
        current_user: Annotated[Dict[str, Any], Depends(get_current_user)],
    ) -> Dict[str, Any]:
        user_role = current_user.get("role", "analyst")
        if user_role not in allowed_roles:
            logger.warning(
                f"Access denied for user '{current_user.get('username')}' "
                f"with role '{user_role}'. Required roles: {allowed_roles}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Insufficient privileges. Role '{user_role}' cannot perform this operation.",
            )
        return current_user

    return role_checker

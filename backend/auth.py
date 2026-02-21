"""
AGNI - Simple JWT-based mock authentication.
"""
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt

# Mock user (no database)
MOCK_USER = {"username": "agni", "password": "farm2025"}
SECRET_KEY = "agni-demo-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


def verify_user(username: str, password: str) -> bool:
    """Verify credentials against mock user."""
    return username == MOCK_USER["username"] and password == MOCK_USER["password"]


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    """Verify JWT and return payload or None."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

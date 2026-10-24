"""
Authentication & User Management Endpoints.

Provides registration, login, JWT token generation, and identity introspection.
"""

from typing import Annotated, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    get_current_user,
    get_user_by_username,
    hash_password,
    save_user,
    verify_password,
)
from app.schemas.auth import TokenResponse, UserLogin, UserRegister, UserResponse
from app.schemas.common import APIResponse

logger = get_logger("routers.auth")

router = APIRouter()


@router.post(
    "/register",
    response_model=APIResponse[UserResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Register New Platform Operator",
    description="Registers a new user account with role-based access control and bcrypt-hashed password.",
)
async def register(payload: UserRegister) -> APIResponse[UserResponse]:
    clean_username = payload.username.strip().lower()
    existing = get_user_by_username(clean_username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with username '{clean_username}' already exists.",
        )

    pwd_hash = hash_password(payload.password)
    user_record = save_user(clean_username, pwd_hash, payload.role.value)
    logger.info(f"Registered operator '{clean_username}' with role '{payload.role.value}'")

    response_data = UserResponse(
        user_id=user_record["user_id"],
        username=user_record["username"],
        role=user_record["role"],
        created_at=user_record["created_at"],
    )
    return APIResponse(success=True, data=response_data, metadata={"message": "User registered successfully"})


@router.post(
    "/login",
    response_model=APIResponse[TokenResponse],
    status_code=status.HTTP_200_OK,
    summary="Operator Authentication & Token Issuance",
    description="Validates operator credentials and issues an HMAC-SHA256 signed JSON Web Token.",
)
async def login(payload: UserLogin) -> APIResponse[TokenResponse]:
    clean_username = payload.username.strip().lower()
    user_record = get_user_by_username(clean_username)
    if not user_record or not verify_password(payload.password, user_record["password_hash"]):
        logger.warning(f"Failed login attempt for username '{clean_username}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        data={"sub": user_record["username"], "role": user_record["role"], "uid": user_record["user_id"]}
    )
    logger.info(f"Operator '{clean_username}' authenticated successfully (Role: {user_record['role']})")

    token_data = TokenResponse(
        access_token=token,
        token_type="bearer",
        username=user_record["username"],
        role=user_record["role"],
        expires_in_minutes=1440,
    )
    return APIResponse(success=True, data=token_data)


@router.get(
    "/me",
    response_model=APIResponse[UserResponse],
    status_code=status.HTTP_200_OK,
    summary="Get Current Authenticated Operator",
    description="Inspects the claims of the current authenticated operator session.",
)
async def get_me(
    current_user: Annotated[Dict[str, Any], Depends(get_current_user)],
) -> APIResponse[UserResponse]:
    response_data = UserResponse(
        user_id=current_user["user_id"],
        username=current_user["username"],
        role=current_user["role"],
        created_at=current_user["created_at"],
    )
    return APIResponse(success=True, data=response_data)

"""Core implementation of Roundtable.ai session validation for FastAPI.

This module provides the Roundtable class, which integrates Roundtable.ai's
risk-based session validation into FastAPI applications as a reusable dependency.
"""

from typing import Annotated, Any, Callable, overload

import aiohttp
from fastapi import HTTPException, params
from pydantic import Field


class Roundtable:
    """A reusable session validator for FastAPI applications using Roundtable.ai.

    An instance represents a configured validator that checks user sessions
    against the Roundtable.ai API based on risk scores. It can be used across
    multiple routes in a FastAPI application to protect endpoints from high-risk
    sessions.
    """

    __slots__ = ("api_key", "status_code", "max_risk_score", "aiohttp_session")
    _base_url = "https://api.roundtable.ai"
    _api_endpoint_url = "/v1/sessions/report"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        status_code: int = 404,
        max_risk_score: int = 50,
    ) -> None:
        """Initialize a session validator with Roundtable.ai API credentials.

        :param api_key: API key for Roundtable.ai authentication. If not provided,
                        reads from ROUNDTABLE_API_KEY environment variable.
        :param status_code: HTTP status code to raise when validation fails.
        :param max_risk_score: Maximum acceptable risk score threshold. Sessions
                               with scores above this value will be rejected.
        :raises ValueError: If api_key is not provided and ROUNDTABLE_API_KEY
                           environment variable is not set.
        """
        if api_key is None:
            from os import environ

            api_key = environ.get("ROUNDTABLE_API_KEY", "")
            if not api_key:
                raise ValueError("ROUNDTABLE_API_KEY environment variable is not set")
        self.api_key = api_key
        self.status_code = status_code
        self.max_risk_score = max_risk_score
        self.aiohttp_session = aiohttp.ClientSession(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
            },
        )

    async def validate_session(self, session_id: str) -> None:
        """Validate a session ID against the Roundtable.ai API.

        Queries the Roundtable.ai API to retrieve the risk score for the given
        session and checks if it exceeds the configured threshold.

        :param session_id: The session identifier to validate.
        :raises HTTPException: If the API request fails or if the session's risk
                              score exceeds the configured max_risk_score.
        """
        async with self.aiohttp_session.get(
            self._api_endpoint_url,
            params={"sessionId": session_id},
        ) as response:
            if response.status != 200:
                raise HTTPException(status_code=self.status_code)
            data = await response.json()
            risk_score = data["risk_score"]
            if risk_score > self.max_risk_score:
                raise HTTPException(status_code=self.status_code)

    @overload
    def __call__(self, *, form_field: str) -> Callable[..., Any]: ...

    @overload
    def __call__(self, *, http_header: str) -> Callable[..., Any]: ...

    @overload
    def __call__(self, *, body_field: str) -> Callable[..., Any]: ...

    def __call__(
        self,
        *,
        form_field: str | None = None,
        http_header: str | None = None,
        body_field: str | None = None,
    ) -> Callable[..., Any]:
        """Create a FastAPI dependency that validates sessions from various sources.

        Returns a dependency function that extracts a session ID from the specified
        source and validates it. Exactly one parameter must be provided.

        :param form_field: Name of the form field containing the session ID.
        :param http_header: Name of the HTTP header containing the session ID.
        :param body_field: Name of the request body field containing the session ID.
        :returns: An async function that can be used as a FastAPI dependency.
        :raises ValueError: If not exactly one parameter is provided.

        Example usage::

            roundtable = Roundtable(api_key="...", status_code=404)

            @app.post("/contact", dependencies=[Depends(roundtable(form_field="rt_session_id"))])
            async def post_contact():
                return {"message": "success"}
        """
        if sum(x is not None for x in (form_field, http_header, body_field)) != 1:
            raise ValueError(
                "Exactly one of form_field, http_header, or body_field must be provided"
            )

        if form_field is not None:
            source = params.Form(alias=form_field)
        elif http_header is not None:
            source = params.Header(alias=http_header)
        elif body_field is not None:
            source = params.Body(alias=body_field)
        else:
            raise ValueError("One of form_field, http_header, or body_field must be provided")

        async def roundtable(
            session_id: Annotated[str, Field(min_length=1), source],
        ) -> None:
            await self.validate_session(session_id)

        return roundtable

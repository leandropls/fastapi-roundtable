"""Core implementation of Roundtable.ai session validation for FastAPI.

This module provides the Roundtable class, which integrates Roundtable.ai's
risk-based session validation into FastAPI applications as a reusable dependency.
"""

import asyncio
from time import perf_counter
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

    __slots__ = (
        "api_key",
        "status_code",
        "max_risk_score",
        "aiohttp_session",
    )
    _base_url = "https://api.roundtable.ai"
    _api_endpoint_url = "/v1/sessions/report"
    _validation_pooling_interval = 1

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

    async def validate_session(
        self,
        session_id: str,
        require_action: str | None = None,
        session_validation_timeout: float = 60,
    ) -> None:
        """Validate a session ID against the Roundtable.ai API.

        Queries the Roundtable.ai API to retrieve the risk score for the given
        session and checks if it exceeds the configured threshold. Optionally
        waits for a specific user action to appear in the session logs.

        :param session_id: The session identifier to validate.
        :param require_action: Optional action name to wait for in session logs.
                              If provided, polls the API until this action appears.
        :param session_validation_timeout: Maximum seconds to wait for the
                                          required action before timing out.
        :raises HTTPException: If the session's risk score exceeds the configured
                              max_risk_score, or if timeout is reached without
                              finding the required action.
        """
        start_time = perf_counter()
        risk_score = 100

        # No need to poll if no action is required, just wait until timeout
        if require_action is None:
            sleep_for = session_validation_timeout
        else:
            sleep_for = self._validation_pooling_interval

        while True:
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            async with self.aiohttp_session.get(
                self._api_endpoint_url,
                params={"sessionId": session_id},
            ) as response:
                if response.status != 200:
                    continue
                data = await response.json()

                if require_action is not None:
                    actions = {item["action"] for item in data["user_logs"]}
                    if require_action in actions:
                        risk_score = data["risk_score"]
                        break

                    if perf_counter() - start_time > session_validation_timeout:
                        # On timeout, if required action not found,
                        # use default risk score
                        break
                else:
                    # If no action is required, accept risk score on timeout
                    risk_score = data["risk_score"]
                    break

        if risk_score > self.max_risk_score:
            raise HTTPException(status_code=self.status_code)

    @overload
    def __call__(
        self,
        *,
        form_field: str,
        require_action: str | None = None,
        session_validation_timeout: float = 30,
    ) -> Callable[..., Any]: ...

    @overload
    def __call__(
        self,
        *,
        http_header: str,
        require_action: str | None = None,
        session_validation_timeout: float = 30,
    ) -> Callable[..., Any]: ...

    @overload
    def __call__(
        self,
        *,
        body_field: str,
        require_action: str | None = None,
        session_validation_timeout: float = 30,
    ) -> Callable[..., Any]: ...

    def __call__(
        self,
        *,
        form_field: str | None = None,
        http_header: str | None = None,
        body_field: str | None = None,
        require_action: str | None = None,
        session_validation_timeout: float = 30,
    ) -> Callable[..., Any]:
        """Create a FastAPI dependency that validates sessions from various sources.

        Returns a dependency function that extracts a session ID from the specified
        source and validates it. Exactly one of form_field, http_header, or
        body_field must be provided.

        :param form_field: Name of the form field containing the session ID.
        :param http_header: Name of the HTTP header containing the session ID.
        :param body_field: Name of the request body field containing the session ID.
        :param require_action: Optional action name to wait for in session logs
                              before validating. Ensures validation includes the
                              complete user behavioral profile.
        :param session_validation_timeout: Maximum seconds to wait for the required
                                          action. Defaults to 30 seconds.
        :returns: An async function that can be used as a FastAPI dependency.
        :raises ValueError: If not exactly one of form_field, http_header, or
                           body_field is provided.

        Example usage::

            roundtable = Roundtable(api_key="...", status_code=404)

            # Basic validation
            @app.post("/contact", dependencies=[Depends(roundtable(form_field="rt_session_id"))])
            async def post_contact():
                return {"message": "success"}

            # Wait for specific action
            @app.post("/submit", dependencies=[Depends(
                roundtable(
                    form_field="rt_session_id",
                    require_action="User submitted form",
                    session_validation_timeout=30
                )
            )])
            async def submit_form():
                return {"status": "ok"}
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
            await self.validate_session(
                session_id=session_id,
                require_action=require_action,
                session_validation_timeout=session_validation_timeout,
            )

        return roundtable

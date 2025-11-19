"""FastAPI integration for Roundtable.ai session validation.

This package provides a FastAPI dependency for validating user sessions
using the Roundtable.ai risk scoring API.
"""

from .main import Roundtable

__all__ = ["Roundtable"]

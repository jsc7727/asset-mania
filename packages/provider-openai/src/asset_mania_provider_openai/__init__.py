"""Optional, approval-gated GPT Image adapter for Asset Mania."""

from .live_transport import HTTPSMultipartTransport
from .turntable import (
    TurntableCallResult,
    build_turntable_prompt,
    build_turntable_request,
    generate_turntable,
)

__all__ = [
    "HTTPSMultipartTransport",
    "TurntableCallResult",
    "build_turntable_prompt",
    "build_turntable_request",
    "generate_turntable",
]

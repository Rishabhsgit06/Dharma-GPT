"""
Compatibility Layer: Redirects all Groq imports to Gemini Client.
Ensures the system uses ONLY Gemini.
"""
from gemini_client import (
    generate_persona_response,
    stream_persona_response,
    transcribe_audio_async,
)

__all__ = [
    "generate_persona_response",
    "stream_persona_response",
    "transcribe_audio_async",
]

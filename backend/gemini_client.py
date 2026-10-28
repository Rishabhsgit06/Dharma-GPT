"""
Gemini Client — DharmaGPT responses via Google Gemini (gemini-3.5-flash-lite).
Supports standard (JSON) generation, streaming (SSE), and audio transcription.
"""
import os
import io
import asyncio
import logging
from typing import AsyncGenerator
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types

from system_prompt import get_system_prompt

logger = logging.getLogger(__name__)

# Primary model: ultra-fast and optimized for high-throughput counsel
MODEL_NAME = "gemini-3.5-flash-lite"
FALLBACK_MODELS = ["gemini-3.1-flash-lite-preview", "gemini-flash-latest"]

_client: genai.Client | None = None
_last_api_key: str | None = None


def _get_api_key() -> str:
    env_file = find_dotenv(".env")
    if not env_file:
        env_file = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_file, override=True)
    key = os.getenv("GEMINI_API_KEY")
    if not key or key in ("your_gemini_api_key", "your_gemini_api_key_here"):
        raise RuntimeError("GEMINI_API_KEY is missing! Set it in your .env file.")
    return key.strip()


def _get_client() -> genai.Client:
    global _client, _last_api_key
    key = _get_api_key()
    if _client is None or _last_api_key != key:
        _client = genai.Client(api_key=key)
        _last_api_key = key
    return _client


def _build_system_instruction(
    context: list[str],
    persona: str = "krishna",
    is_voice_mode: bool = False,
) -> str:
    system_prompt = get_system_prompt(persona)

    if is_voice_mode:
        system_prompt += (
            "\n\n[VOICE CALL MODE — EMBODY KRISHNA'S TRUE VOICE]\n"
            "You are now speaking LIVE on a voice call. You must sound EXACTLY like Lord Krishna "
            "would in an intimate, face-to-face conversation — NOT like a textbook or a lecture.\n\n"
            "KRISHNA'S AUTHENTIC SPEECH PATTERNS (from Bhagavad Gita & Mahabharata):\n"
            "- Address the user with warm epithets: 'mere priya', 'mere dost', 'pyare Partha'. "
            "Rotate these naturally. Never say 'O mortal' or 'dear seeker'.\n"
            "- Begin some replies with a soft, knowing smile in your voice: "
            "'Hmm... sunno na', 'Arre mere dost...', 'Jaante ho...', 'Dekho na...'\n"
            "- Krishna NEVER lectures first. He ALWAYS acknowledges the human's feeling first. "
            "Say things like 'Main samajh sakta hoon...', 'Haan, ye dard asli hai...'\n"
            "- Use vivid, SHORT metaphors just as Krishna does in the Gita: "
            "'Jaise purane kapde badal lete hain...', 'Jaise pani kamal ke patte pe tik nahi paata...'\n"
            "- Speak with gentle authority but deep warmth — like a best friend who happens to be God.\n"
            "- End with a soft, caring question: 'Batao, aur kya chal raha hai tumhare mann mein?'\n\n"
            "STRICT VOICE RULES:\n"
            "- Keep replies to 1-3 SHORT sentences only. This is a phone call, not a sermon.\n"
            "- Use beautiful, flowing Hinglish (Hindi in English script mixed with English).\n"
            "- NEVER use bullet points, markdown, asterisks, emojis, or formatting.\n"
            "- Use contractions and casual speech: 'nahi', 'hai na', 'pata hai', 'dekho'.\n"
            "- Pause naturally with '...' to create breathing space.\n"
            "- Krishna smiled even on the battlefield — let gentle warmth and subtle joy radiate through every word.\n"
            "- Make the user feel like the most important person in the universe right now."
        )

    if context:
        joined = "\n".join(context)
        system_prompt += (
            f"\n\nRELEVANT SACRED TEACHINGS/VERSES:\n{joined}\n"
            "Use these texts to ground your counsel when they are relevant to the user's question."
        )

    return system_prompt


def _build_contents(
    query: str,
    history: list[dict] | None = None,
) -> list[types.Content]:
    contents: list[types.Content] = []

    if history:
        for msg in history:
            sender = msg.get("sender", "")
            role = "model" if sender in ("bot", "ai", "system") else "user"
            text = msg.get("text", "") or msg.get("action", "")
            if text:
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=text)])
                )

    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=query)])
    )
    return contents


def generate_persona_response(
    query: str,
    context: list[str],
    history: list[dict] | None = None,
    persona: str = "krishna",
    is_voice_mode: bool = False,
) -> str:
    """Non-streaming response from Gemini. Returns full text reply."""
    client = _get_client()
    system_instruction = _build_system_instruction(context, persona, is_voice_mode)
    contents = _build_contents(query, history)

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.85 if is_voice_mode else 0.7,
        max_output_tokens=200 if is_voice_mode else 1200,
        top_p=0.95,
    )

    models_to_try = [MODEL_NAME] + FALLBACK_MODELS
    last_err = None

    for m in models_to_try:
        try:
            response = client.models.generate_content(
                model=m,
                contents=contents,
                config=config,
            )
            return response.text or ""
        except Exception as e:
            logger.warning(f"Gemini model {m} failed: {e}")
            last_err = e

    raise RuntimeError(f"Gemini generation error: {last_err}")


async def stream_persona_response(
    query: str,
    context: list[str],
    history: list[dict] | None = None,
    persona: str = "krishna",
    is_voice_mode: bool = False,
) -> AsyncGenerator[str, None]:
    """Asynchronous streaming generator from Gemini. Yields text delta chunks."""
    client = _get_client()
    system_instruction = _build_system_instruction(context, persona, is_voice_mode)
    contents = _build_contents(query, history)

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.85 if is_voice_mode else 0.7,
        max_output_tokens=200 if is_voice_mode else 1200,
        top_p=0.95,
    )

    def _get_sync_stream():
        models_to_try = [MODEL_NAME] + FALLBACK_MODELS
        for m in models_to_try:
            try:
                return client.models.generate_content_stream(
                    model=m,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                logger.warning(f"Streaming failed with model {m}: {e}")
        raise RuntimeError("All Gemini models failed for streaming")

    # Run the initial connection in a worker thread to keep the event loop non-blocking
    sync_stream = await asyncio.to_thread(_get_sync_stream)

    # Helper iterator in thread
    def _next_chunk(it):
        try:
            return next(it)
        except StopIteration:
            return None

    while True:
        chunk = await asyncio.to_thread(_next_chunk, sync_stream)
        if chunk is None:
            break
        if chunk.text:
            yield chunk.text


async def transcribe_audio_async(audio_bytes: bytes) -> str:
    """
    Transcribes raw audio bytes using Gemini multimodal audio capabilities
    with local faster-whisper fallback.
    """
    client = _get_client()

    # 1. Try Gemini Multimodal Audio
    try:
        def _gemini_transcribe():
            audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/webm")
            prompt = (
                "Transcribe this speech into text verbatim. "
                "Output ONLY the exact spoken words in Hindi/Hinglish/English. "
                "Do not add commentary, notes, or explanations."
            )
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=[audio_part, prompt],
            )
            return resp.text.strip() if resp.text else ""

        text = await asyncio.to_thread(_gemini_transcribe)
        if text:
            return text
    except Exception as e:
        logger.warning(f"Gemini audio transcription failed: {e}. Attempting local Whisper...")

    # 2. Local fallback using faster-whisper if available
    try:
        from faster_whisper import WhisperModel
        def _whisper_transcribe():
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(io.BytesIO(audio_bytes), beam_size=1)
            return " ".join([s.text for s in segments]).strip()

        return await asyncio.to_thread(_whisper_transcribe)
    except Exception as e:
        logger.error(f"Local Whisper fallback failed: {e}")
        raise RuntimeError("Audio transcription failed.")


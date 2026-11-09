import os
import tempfile
import wave
from io import BytesIO

from groq import Groq
from config.settings import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

STT_MODEL = "whisper-large-v3-turbo"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"


def _guess_audio_filename(audio_bytes: bytes, audio_mime_type=None) -> str:
    """Return a filename with an extension matching the recorded audio."""
    mime = (audio_mime_type or "").lower().split(";")[0].strip()

    mime_extensions = {
        "audio/webm": "voice_input.webm",
        "video/webm": "voice_input.webm",
        "audio/ogg": "voice_input.ogg",
        "application/ogg": "voice_input.ogg",
        "audio/mp4": "voice_input.mp4",
        "video/mp4": "voice_input.mp4",
        "audio/m4a": "voice_input.m4a",
        "audio/wav": "voice_input.wav",
        "audio/x-wav": "voice_input.wav",
        "audio/mpeg": "voice_input.mp3",
        "audio/mp3": "voice_input.mp3",
        "audio/flac": "voice_input.flac",
    }
    if mime in mime_extensions:
        return mime_extensions[mime]

    # Detect common file signatures when the browser doesn't send a MIME type.
    if audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE":
        return "voice_input.wav"
    if audio_bytes.startswith(b"OggS"):
        return "voice_input.ogg"
    if audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
        return "voice_input.webm"
    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "voice_input.mp3"
    if len(audio_bytes) > 12 and audio_bytes[4:8] == b"ftyp":
        return "voice_input.mp4"
    if audio_bytes.startswith(b"fLaC"):
        return "voice_input.flac"

    # Browser MediaRecorder commonly uses WebM even when type metadata is absent.
    raise ValueError(
        "Unsupported or unknown audio format. Please record again using the microphone button."
    )


def transcribe_audio(audio_bytes, language=None, audio_mime_type=None):
    """Transcribe recorded audio with Groq Whisper and return recognized text."""
    if not audio_bytes:
        return ""
    if not isinstance(audio_bytes, (bytes, bytearray)):
        raise TypeError("Audio input must be bytes.")

    audio_bytes = bytes(audio_bytes)
    if len(audio_bytes) < 100:
        raise ValueError("The recording is too short or empty. Please record again.")

    filename = _guess_audio_filename(audio_bytes, audio_mime_type)
    request = {
        "file": (filename, BytesIO(audio_bytes)),
        "model": STT_MODEL,
        "response_format": "json",
        "temperature": 0.0,
    }
    if language in ("en", "ta"):
        request["language"] = language

    result = client.audio.transcriptions.create(**request)
    return (getattr(result, "text", "") or "").strip()


def _split_text_for_tts(text, max_characters=190):
    """Split long text into manageable chunks without cutting words where possible."""
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    while len(text) > max_characters:
        split_at = text.rfind(" ", 0, max_characters)
        if split_at < max_characters // 2:
            split_at = max_characters
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


def _request_tts_audio(text):
    """Request one WAV audio segment from Groq's speech generation endpoint."""
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    )
    # SDK versions differ: newer responses expose stream_to_file; some expose content.
    if hasattr(response, "read"):
        return response.read()
    if hasattr(response, "content") and response.content:
        return response.content
    if hasattr(response, "stream_to_file"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            temp_path = temp.name
        try:
            response.stream_to_file(temp_path)
            with open(temp_path, "rb") as audio_file:
                return audio_file.read()
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
    raise RuntimeError("The speech service returned no audio data.")


def text_to_speech(text):
    """Convert text to WAV audio bytes; returns None for empty text."""
    chunks = _split_text_for_tts(text)
    if not chunks:
        return None

    wav_segments = []
    for chunk in chunks:
        audio_bytes = _request_tts_audio(chunk)
        if not audio_bytes:
            continue
        try:
            with wave.open(BytesIO(audio_bytes), "rb") as wav_file:
                params = wav_file.getparams()
                frames = wav_file.readframes(wav_file.getnframes())
        except (wave.Error, EOFError) as error:
            raise ValueError(
                "The voice service did not return a valid WAV file."
            ) from error
        wav_segments.append((params, frames))

    if not wav_segments:
        return None

    first_params = wav_segments[0][0]
    output = BytesIO()
    with wave.open(output, "wb") as combined:
        combined.setnchannels(first_params.nchannels)
        combined.setsampwidth(first_params.sampwidth)
        combined.setframerate(first_params.framerate)
        for params, frames in wav_segments:
            if (
                params.nchannels != first_params.nchannels
                or params.sampwidth != first_params.sampwidth
                or params.framerate != first_params.framerate
                or params.comptype != first_params.comptype
            ):
                raise ValueError("Voice audio segments use incompatible WAV formats.")
            combined.writeframes(frames)
    return output.getvalue()

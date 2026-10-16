from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # service
    app_name: str = "voice-agent-twilio"
    environment: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    public_base_url: str = ""
    log_level: str = "INFO"
    session_ttl_s: int = 1800
    session_cleanup_interval_s: int = 60

    # twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    transfer_number: str = ""
    transfer_message: str = "One moment while I connect you to a specialist."
    dial_timeout_s: int = 30

    # identity / persona
    business_name: str = "Acme Store"
    agent_name: str = "Nova"
    greeting: str = "Thanks for calling Acme support. How can I help you today?"

    # providers
    groq_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""

    llm_provider: str = "litellm"  # litellm | mock
    llm_model: str = "groq/llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 400
    llm_timeout_s: float = 20.0
    max_tool_iterations: int = 4

    stt_provider: str = "groq"  # groq | openai | mock
    stt_model: str = "whisper-large-v3-turbo"
    stt_language: str = "en"
    stt_timeout_s: float = 25.0
    stt_no_speech_max: float = 0.75

    tts_provider: str = "groq"  # groq | openai | mock
    tts_model: str = "playai-tts"
    tts_voice: str = "Fritz-PlayAI"
    tts_timeout_s: float = 25.0
    tts_fallback_tone: bool = True
    tts_max_chars: int = 400

    # rag
    embeddings_provider: str = "auto"  # auto | fastembed | hashing | none
    embeddings_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str = str(ROOT_DIR / ".cache" / "embeddings")
    kb_path: str = str(ROOT_DIR / "knowledge")
    rag_top_k: int = 4
    rag_min_score: float | None = None  # None -> calibrated per embedding provider
    rag_chunk_chars: int = 700
    rag_chunk_overlap: int = 120

    # conversation policy
    confidence_threshold: float = 0.6
    handoff_digit: str = "0"
    handoff_ack: str = "Of course. Let me get someone on the line for you."
    repeat_prompt: str = "Sorry, I didn't catch that. Could you say it again?"
    max_history_messages: int = 20

    # vad / endpointing
    frame_ms: int = 20
    vad_silence_ms: int = 700
    vad_min_speech_ms: int = 200
    vad_start_rms: int = 450
    vad_end_rms: int = 300
    vad_barge_frames: int = 3
    max_utterance_s: float = 20.0
    max_buffer_s: float = 30.0

    # integrations
    crm_base_url: str = "http://localhost:8001"
    crm_api_key: str = "dev-crm-key"
    crm_timeout_s: float = 8.0
    n8n_webhook_url: str = ""
    summarize_calls: bool = False

    # observability
    metrics_window: int = 500

    def resolved_public_base(self) -> str:
        return self.public_base_url.rstrip("/")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

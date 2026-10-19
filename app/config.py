"""Application configuration using Pydantic Settings."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    llm_provider: Literal["groq", "gemini", "openai"] = Field(
        default="groq", description="LLM provider name"
    )
    llm_model: str = Field(
        default="openai/gpt-oss-120b", description="Model identifier"
    )
    temperature: float = Field(default=0.0, description="Sampling temperature (always 0 for eval)")

    # API Keys
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini / Google API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    # Cache Settings
    llm_cache_enabled: bool = Field(default=True, description="Enable disk cache for LLM calls")
    llm_cache_dir: Path = Field(
        default=Path("./cache/llm_cache"), description="Directory storing LLM cache files"
    )

    # Rate Limiting & Concurrency
    llm_concurrency: int = Field(default=2, description="Maximum concurrent LLM requests")
    llm_max_retries: int = Field(default=15, description="Maximum retries with exponential backoff")
    llm_retry_min_wait: float = Field(default=1.0, description="Minimum backoff wait in seconds")
    llm_retry_max_wait: float = Field(default=60.0, description="Maximum backoff wait in seconds")

    # LangSmith Tracing
    langsmith_tracing: bool = Field(default=False, description="Enable LangSmith tracing")
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    langsmith_project: str = Field(default="text-to-sql-agent", description="LangSmith project name")
    langsmith_endpoint: Optional[str] = Field(default=None, description="LangSmith endpoint")

    # Database & Execution Sandbox Settings
    query_timeout_s: float = Field(default=10.0, description="Default query execution timeout in seconds")
    eval_query_timeout_s: float = Field(default=30.0, description="Evaluation query timeout in seconds")
    max_returned_rows: int = Field(default=200, description="Cap on fetched rows for UI/API")

    # Directory Paths
    data_dir: Path = Field(default=Path("./data"), description="Base datasets directory")
    demo_data_dir: Path = Field(default=Path("./demo_data"), description="Demo databases directory")
    eval_dir: Path = Field(default=Path("./eval"), description="Evaluation outputs directory")
    chroma_dir: Path = Field(default=Path("./data/chroma"), description="Persistent Chroma directory")
    demo_chroma_dir: Path = Field(default=Path("./demo_data/chroma"), description="Demo Chroma directory")


settings = Settings()


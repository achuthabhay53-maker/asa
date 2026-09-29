"""Runtime settings loaded from env (pydantic-settings)."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aca_env: str = "dev"
    aca_port: int = 8002
    aca_log_level: str = "INFO"

    database_url: str

    aimpact_base_url: str
    aimpact_api_token: str
    aimpact_timeout_s: int = 30

    anthropic_api_key: str
    anthropic_model_haiku:  str = "claude-haiku-4-5-20251001"
    anthropic_model_sonnet: str = "claude-sonnet-5"

    aws_region: str = "ap-south-1"
    aws_kms_key_id: str | None = None

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    aca_daily_usd_cap_per_tenant: float = 25.0
    aca_job_timeout_s: int = 900
    aca_max_revise_loops: int = 2


@lru_cache
def settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

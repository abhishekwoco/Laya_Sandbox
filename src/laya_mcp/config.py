"""Server configuration, read from LAYA_MCP_* environment variables or a .env file."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LAYA_MCP_", env_file=".env", extra="ignore")

    # network
    host: str = "0.0.0.0"
    port: int = 8765

    # storage
    data_dir: Path = Path("data")

    # inference
    device: str | None = None               # None = laya picks (cpu on this host)
    threads: int | None = 6                 # torch intra-op threads; keep <= physical cores
    preload: str = "english"                # comma list of checkpoints loaded at startup
    max_loaded: int = 2                     # LRU cap on resident checkpoints
    warmup: bool = True                     # run one tiny prediction after preload
    quantize: bool = False                  # int8 dynamic quantization: rejected, changes 26% of answers (docs/perf.md)

    # request limits
    sync_row_budget: int = 20               # items x questions allowed in one synchronous call (~9-32 s, docs/perf.md)
    max_state_chars: int = 20_000
    max_questions: int = 20
    max_items_per_call: int = 500           # per laya_classify_batch call (clients append in chunks)
    request_timeout_s: float = 120.0
    max_waiting_requests: int = 4           # interactive calls queued behind the inference lock; more get a retryable EngineBusy

    # decision policy defaults
    default_threshold: float = 0.8
    default_target_accuracy: float = 0.9
    min_eval_examples: int = 50

    # jobs
    job_workers: int = 1

    # identity / logging
    default_team: str = "default"
    team_header: str = "x-laya-team"
    log_content: bool = False               # never log raw state text unless explicitly enabled
    log_level: str = "INFO"

    @property
    def preload_models(self) -> list[str]:
        return [m.strip() for m in self.preload.split(",") if m.strip()]

    @property
    def db_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'laya_mcp.db').as_posix()}"

    @property
    def jobs_db_path(self) -> Path:
        return self.data_dir / "jobs.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()

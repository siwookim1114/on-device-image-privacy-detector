from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # CORS
    frontend_url: str = "http://localhost:5173"

    # MongoDB
    mongo_url: str = "mongodb://localhost:27017"
    mongo_db: str = "privacy_guard"

    # Pipeline
    pipeline_config_path: str = "../configs/config.yaml"
    upload_dir: str = "../data/tmp_uploads"
    results_dir: str = "../data/full_pipeline_results"

    # Session
    session_ttl_hours: int = 8
    max_concurrent_pipelines: int = 2

    # VLM
    vlm_health_url: str = "http://localhost:8081/health"

    class Config:
        env_prefix = "PRIVACY_"
        env_file = ".env"


settings = Settings()

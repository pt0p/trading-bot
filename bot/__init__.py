"""Самодостаточный production-модуль для торгового пайплайна."""

from bot.main import PipelineConfig, PipelineResult, ProductionPipeline, run_pipeline
from bot.telegram_bot import (
    TradingTelegramBot,
    UserValidator,
    WebhookServerConfig,
    parse_webhook_config_from_env,
    trading_bot_from_env,
)

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "ProductionPipeline",
    "TradingTelegramBot",
    "UserValidator",
    "WebhookServerConfig",
    "parse_webhook_config_from_env",
    "run_pipeline",
    "trading_bot_from_env",
]

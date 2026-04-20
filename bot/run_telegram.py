"""CLI-точка входа: запуск Telegram-бота в режиме webhook."""

from __future__ import annotations

import logging
import sys

from bot.telegram_bot import parse_webhook_config_from_env, trading_bot_from_env


def main() -> None:
    """Читает переменные окружения и запускает webhook-сервер Telegram-бота.

    Обязательные переменные окружения:

    - ``TELEGRAM_BOT_TOKEN`` — токен бота;
    - ``WEBHOOK_URL`` — полный HTTPS URL для ``setWebhook`` (путь задаётся вместе с хостом);
    - ``WEBHOOK_LISTEN_PORT`` — локальный порт процесса (целое число).

    Дополнительно:

    - ``ALLOWED_TELEGRAM_USER_IDS`` — whitelist ``user_id`` через запятую;
    - ``DEFAULT_SECURITY`` — тикер (по умолчанию ``SBER``);
    - ``ARTIFACTS_DIR`` — каталог артефактов (по умолчанию ``models``);
    - ``WEBHOOK_LISTEN_HOST`` — адрес привязки (по умолчанию ``0.0.0.0``);
    - ``WEBHOOK_SECRET_TOKEN`` — секрет для заголовка Telegram (опционально);
    - ``WEBHOOK_CERT_PATH``, ``WEBHOOK_KEY_PATH`` — PEM для HTTPS на процессе (опционально, пара).

    Raises
    ------
    SystemExit
        Если не заданы обязательные переменные или конфигурация некорректна.
    """

    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    try:
        bot = trading_bot_from_env()
        webhook_config = parse_webhook_config_from_env()
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    bot.run_webhook_sync(webhook_config)


if __name__ == "__main__":
    main()

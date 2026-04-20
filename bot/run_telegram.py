"""CLI-точка входа: запуск Telegram-бота поверх production-пайплайна."""

from __future__ import annotations

import logging
import sys

from bot.telegram_bot import trading_bot_from_env


def main() -> None:
    """Читает переменные окружения и запускает long polling Telegram-бота.

    Ожидаемые переменные окружения:

    - ``TELEGRAM_BOT_TOKEN`` — токен бота (обязательно);
    - ``ALLOWED_TELEGRAM_USER_IDS`` — список разрешённых ``user_id`` через запятую;
    - ``DEFAULT_SECURITY`` — тикер инструмента (по умолчанию ``SBER``);
    - ``ARTIFACTS_DIR`` — каталог артефактов моделей (по умолчанию ``models``).

    Raises
    ------
    SystemExit
        Если не задан ``TELEGRAM_BOT_TOKEN`` или при фатальной ошибке запуска.
    """

    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    try:
        bot = trading_bot_from_env()
    except ValueError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    bot.run_polling_sync()


if __name__ == "__main__":
    main()

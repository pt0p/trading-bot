"""Telegram-обёртка над production-пайплайном: валидация пользователя и запуск расчёта."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from bot.main import PipelineResult, run_pipeline

logger = logging.getLogger(__name__)


def _require_user_data(context: ContextTypes.DEFAULT_TYPE) -> dict[Any, Any]:
    """Возвращает словарь ``user_data`` или завершает работу с ошибкой.

    Parameters
    ----------
    context : ContextTypes.DEFAULT_TYPE
        Контекст PTB.

    Returns
    -------
    dict[Any, Any]
        Изменяемый словарь данных пользователя.

    Raises
    ------
    RuntimeError
        Если ``user_data`` отсутствует (неожиданно для ``ConversationHandler``).
    """

    data = context.user_data
    if data is None:
        msg = "context.user_data отсутствует в ConversationHandler"
        raise RuntimeError(msg)
    return data


ASK_START_DATE: Final[int] = 1
ASK_END_DATE: Final[int] = 2
ASK_CAPITAL: Final[int] = 3


@dataclass(frozen=True, slots=True)
class WebhookServerConfig:
    """Параметры HTTP-сервера и webhook для ``Application.run_webhook``.

    Parameters
    ----------
    listen : str
        Адрес привязки сокета (например ``0.0.0.0`` или ``127.0.0.1``).
    port : int
        Порт локального сервера, на который проксируются запросы Telegram.
    url_path : str
        Относительный путь без начального слэша (совпадает с путём из публичного ``webhook_url``).
    webhook_url : str
        Полный HTTPS URL, передаваемый в Telegram ``setWebhook``.
    secret_token : str or None
        Опциональный секретный токен для заголовка ``X-Telegram-Bot-Api-Secret-Token``.
    cert : pathlib.Path or None
        Опциональный файл сертификата TLS для HTTPS на процессе.
    key : pathlib.Path or None
        Опциональный файл ключа TLS для HTTPS на процессе.
    """

    listen: str
    port: int
    url_path: str
    webhook_url: str
    secret_token: str | None = None
    cert: Path | None = None
    key: Path | None = None


def parse_webhook_config_from_env() -> WebhookServerConfig:
    """Читает конфигурацию webhook из переменных окружения.

    Ожидается:

    - ``WEBHOOK_URL`` — полный публичный URL для ``setWebhook`` (обязательно);
    - ``WEBHOOK_LISTEN_HOST`` — хост привязки (по умолчанию ``0.0.0.0``);
    - ``WEBHOOK_LISTEN_PORT`` — порт процесса (обязательно);
    - ``WEBHOOK_SECRET_TOKEN`` — необязательный секрет Telegram;
    - ``WEBHOOK_CERT_PATH``, ``WEBHOOK_KEY_PATH`` — необязательные пути к PEM для TLS на процессе
      (если задан один, нужен и второй).

    Returns
    -------
    WebhookServerConfig
        Готовые параметры для ``Application.run_webhook``.

    Raises
    ------
    ValueError
        Если отсутствует обязательное поле или комбинация параметров некорректна.
    """

    raw_webhook_url = os.environ.get("WEBHOOK_URL", "").strip()
    if not raw_webhook_url:
        msg = "Переменная окружения WEBHOOK_URL не задана или пуста."
        raise ValueError(msg)

    parsed_url = urlparse(raw_webhook_url)
    if parsed_url.scheme.lower() != "https":
        msg = "WEBHOOK_URL должен использовать схему https (требование Telegram Bot API)."
        raise ValueError(msg)

    raw_path = (parsed_url.path or "/").strip("/")
    url_path = raw_path if raw_path else "webhook"

    listen_host = os.environ.get("WEBHOOK_LISTEN_HOST", "0.0.0.0").strip()
    if not listen_host:
        msg = "WEBHOOK_LISTEN_HOST не может быть пустым."
        raise ValueError(msg)

    port_raw = os.environ.get("WEBHOOK_LISTEN_PORT", "").strip()
    if not port_raw:
        msg = "Переменная окружения WEBHOOK_LISTEN_PORT обязательна (целое число)."
        raise ValueError(msg)
    try:
        listen_port = int(port_raw)
    except ValueError as exc:
        msg = f"WEBHOOK_LISTEN_PORT должно быть целым числом, получено {port_raw!r}."
        raise ValueError(msg) from exc
    if not (1 <= listen_port <= 65535):
        msg = "WEBHOOK_LISTEN_PORT должно быть в диапазоне 1–65535."
        raise ValueError(msg)

    secret_raw = os.environ.get("WEBHOOK_SECRET_TOKEN")
    secret_token = secret_raw.strip() if secret_raw and secret_raw.strip() else None

    cert_raw = os.environ.get("WEBHOOK_CERT_PATH", "").strip()
    key_raw = os.environ.get("WEBHOOK_KEY_PATH", "").strip()
    cert_path: Path | None = Path(cert_raw).resolve() if cert_raw else None
    key_path: Path | None = Path(key_raw).resolve() if key_raw else None

    if (cert_path is None) ^ (key_path is None):
        msg = "Задайте оба параметра WEBHOOK_CERT_PATH и WEBHOOK_KEY_PATH или ни одного."
        raise ValueError(msg)

    return WebhookServerConfig(
        listen=listen_host,
        port=listen_port,
        url_path=url_path,
        webhook_url=raw_webhook_url,
        secret_token=secret_token,
        cert=cert_path,
        key=key_path,
    )


@dataclass(frozen=True, slots=True)
class UserValidator:
    """Проверяет, разрешён ли доступ для Telegram-пользователя по whitelist.

    Parameters
    ----------
    allowed_user_ids : frozenset[int]
        Множество разрешённых числовых идентификаторов пользователей Telegram.
    """

    allowed_user_ids: frozenset[int]

    @classmethod
    def from_env_value(cls, raw_value: str | None) -> UserValidator:
        """Создаёт валидатор из строки переменной окружения ``ALLOWED_TELEGRAM_USER_IDS``.

        Parameters
        ----------
        raw_value : str or None
            Строка вида ``"123,456"`` или ``None``/пустая строка (ни одного разрешённого id).

        Returns
        -------
        UserValidator
            Экземпляр с разобранным whitelist.
        """

        if not raw_value or not str(raw_value).strip():
            return cls(allowed_user_ids=frozenset())

        parts = [part.strip() for part in str(raw_value).split(",")]
        ids: set[int] = set()
        for part in parts:
            if not part:
                continue
            ids.add(int(part))
        return cls(allowed_user_ids=frozenset(ids))

    def is_allowed(self, user_id: int | None) -> bool:
        """Возвращает ``True``, если пользователь есть в whitelist.

        Parameters
        ----------
        user_id : int or None
            Идентификатор пользователя Telegram.

        Returns
        -------
        bool
            ``True`` если доступ разрешён; ``False`` если ``user_id`` отсутствует
            или не входит в whitelist.
        """

        if user_id is None:
            return False
        if not self.allowed_user_ids:
            return False
        return int(user_id) in self.allowed_user_ids


def parse_user_date(text: str) -> date:
    """Разбирает дату из текста пользователя в одном из поддерживаемых форматов.

    Поддерживаемые форматы:
    - ``%d.%m.%Y`` (например, ``20.04.2026``);
    - ``%Y-%m-%d`` (например, ``2026-04-20``).

    Parameters
    ----------
    text : str
        Строка с датой.

    Returns
    -------
    date
        Разобранная календарная дата.

    Raises
    ------
    ValueError
        Если строка не соответствует ни одному из поддерживаемых форматов.
    """

    raw = text.strip()
    if not raw:
        msg = "Пустая строка даты"
        raise ValueError(msg)

    try:
        return date.fromisoformat(raw)
    except ValueError:
        pass

    try:
        return datetime.strptime(raw, "%d.%m.%Y").date()  # noqa: DTZ007
    except ValueError:
        pass

    msg = (
        "Неверный формат даты. Укажите дату как ДД.ММ.ГГГГ "
        "(например, 20.04.2026) или ГГГГ-ММ-ДД (например, 2026-04-20)."
    )
    raise ValueError(msg)


def _format_iso(d: date) -> str:
    """Преобразует ``date`` в строку ``YYYY-MM-DD`` для пайплайна.

    Parameters
    ----------
    d : date
        Дата.

    Returns
    -------
    str
        Строка в формате ISO.
    """

    return d.isoformat()


class TradingTelegramBot:
    """Инициализирует Telegram-приложение и регистрирует команды/диалог расчёта."""

    def __init__(
        self,
        *,
        token: str,
        validator: UserValidator,
        default_security: str = "SBER",
        artifacts_dir: str | Path = "models",
    ) -> None:
        """Создаёт конфигурацию бота.

        Parameters
        ----------
        token : str
            Токен Telegram Bot API.
        validator : UserValidator
            Валидатор доступа по whitelist ``user_id``.
        default_security : str, default="SBER"
            Тикер инструмента для загрузки истории по умолчанию.
        artifacts_dir : str or Path, default="models"
            Каталог с артефактами моделей.
        """

        self._token = token
        self._validator = validator
        self._default_security = default_security.strip().upper() or "SBER"
        self._artifacts_dir = Path(artifacts_dir)

    def build_application(self) -> Application:
        """Собирает ``Application`` с зарегистрированными хэндлерами.

        Returns
        -------
        Application
            Готовое приложение python-telegram-bot.
        """

        application = Application.builder().token(self._token).build()
        application.add_handler(CommandHandler("start", self._cmd_start))
        application.add_handler(CommandHandler("validate", self._cmd_validate))

        run_conversation = ConversationHandler(
            entry_points=[CommandHandler("run", self._cmd_run_entry)],
            states={
                ASK_START_DATE: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_start_date),
                ],
                ASK_END_DATE: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_end_date),
                ],
                ASK_CAPITAL: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_capital),
                ],
            },
            fallbacks=[
                CommandHandler("cancel", self._cmd_cancel),
            ],
            name="run_pipeline_conversation",
            persistent=False,
        )
        application.add_handler(run_conversation)
        return application

    def run_webhook_sync(self, webhook_config: WebhookServerConfig) -> None:
        """Запускает бота в режиме webhook (HTTP-сервер и вызов ``setWebhook``).

        Parameters
        ----------
        webhook_config : WebhookServerConfig
            Параметры привязки и URL, передаваемые в ``Application.run_webhook``.
        """

        application = self.build_application()
        application.run_webhook(
            listen=webhook_config.listen,
            port=webhook_config.port,
            url_path=webhook_config.url_path,
            webhook_url=webhook_config.webhook_url,
            allowed_updates=Update.ALL_TYPES,
            secret_token=webhook_config.secret_token,
            cert=str(webhook_config.cert) if webhook_config.cert is not None else None,
            key=str(webhook_config.key) if webhook_config.key is not None else None,
        )

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обрабатывает команду ``/start``: краткая справка по командам.

        Parameters
        ----------
        update : Update
            Входящее обновление Telegram.
        context : ContextTypes.DEFAULT_TYPE
            Контекст выполнения PTB.
        """

        if update.message is None:
            return

        text = (
            "Доступные команды:\n"
            "/validate — проверить доступ по whitelist\n"
            "/run — запустить расчёт (даты начала/конца и начальный капитал)\n"
            "/cancel — отменить текущий диалог /run"
        )
        await update.message.reply_text(text)

    async def _cmd_validate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обрабатывает команду ``/validate``: статус доступа пользователя.

        Parameters
        ----------
        update : Update
            Входящее обновление Telegram.
        context : ContextTypes.DEFAULT_TYPE
            Контекст выполнения PTB.
        """

        if update.message is None:
            return

        user = update.effective_user
        allowed = self._validator.is_allowed(user.id if user else None)
        if allowed:
            await update.message.reply_text(
                f"Доступ разрешён (user_id={user.id if user else 'unknown'})."
            )
            return

        await update.message.reply_text(
            "Доступ запрещён: ваш Telegram user_id отсутствует в whitelist."
        )

    async def _cmd_run_entry(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Начинает диалог ``/run``: проверка доступа и запрос даты начала.

        Parameters
        ----------
        update : Update
            Входящее обновление Telegram.
        context : ContextTypes.DEFAULT_TYPE
            Контекст выполнения PTB.

        Returns
        -------
        int
            Следующее состояние ConversationHandler или ``ConversationHandler.END``.
        """

        if update.message is None:
            return ConversationHandler.END

        user = update.effective_user
        if not self._validator.is_allowed(user.id if user else None):
            await update.message.reply_text(
                "Доступ запрещён: ваш Telegram user_id отсутствует в whitelist."
            )
            return ConversationHandler.END

        try:
            user_data = _require_user_data(context)
        except RuntimeError:
            await update.message.reply_text("Не удалось инициализировать состояние диалога. Попробуйте /run ещё раз.")
            return ConversationHandler.END

        user_data.pop("start_date", None)
        user_data.pop("end_date", None)

        await update.message.reply_text(
            "Укажите дату начала периода в формате ДД.ММ.ГГГГ или ГГГГ-ММ-ДД:"
        )
        return ASK_START_DATE

    async def _on_start_date(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Принимает и парсит дату начала периода.

        Parameters
        ----------
        update : Update
            Входящее обновление Telegram.
        context : ContextTypes.DEFAULT_TYPE
            Контекст выполнения PTB.

        Returns
        -------
        int
            Следующее состояние ConversationHandler.
        """

        if update.message is None or update.message.text is None:
            return ASK_START_DATE

        try:
            user_data = _require_user_data(context)
        except RuntimeError:
            await update.message.reply_text("Внутренняя ошибка состояния. Начните снова с /run.")
            return ConversationHandler.END

        try:
            start = parse_user_date(update.message.text)
        except ValueError as exc:
            await update.message.reply_text(str(exc))
            return ASK_START_DATE

        user_data["start_date"] = start
        await update.message.reply_text(
            "Укажите дату конца периода в формате ДД.ММ.ГГГГ или ГГГГ-ММ-ДД:"
        )
        return ASK_END_DATE

    async def _on_end_date(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Принимает дату конца и проверяет порядок дат.

        Parameters
        ----------
        update : Update
            Входящее обновление Telegram.
        context : ContextTypes.DEFAULT_TYPE
            Контекст выполнения PTB.

        Returns
        -------
        int
            Следующее состояние ConversationHandler.
        """

        if update.message is None or update.message.text is None:
            return ASK_END_DATE

        try:
            user_data = _require_user_data(context)
        except RuntimeError:
            await update.message.reply_text("Внутренняя ошибка состояния. Начните снова с /run.")
            return ConversationHandler.END

        start_date = user_data.get("start_date")
        if not isinstance(start_date, date):
            await update.message.reply_text("Внутренняя ошибка: не найдена дата начала. Начните снова с /run.")
            return ConversationHandler.END

        try:
            end = parse_user_date(update.message.text)
        except ValueError as exc:
            await update.message.reply_text(str(exc))
            return ASK_END_DATE

        if start_date > end:
            await update.message.reply_text(
                "Дата начала не может быть позже даты конца. Введите дату конца ещё раз:"
            )
            return ASK_END_DATE

        user_data["end_date"] = end
        await update.message.reply_text(
            "Укажите начальный капитал в рублях (положительное число, например 100000):"
        )
        return ASK_CAPITAL

    async def _on_capital(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Принимает начальный капитал, запускает пайплайн и отправляет графики и сводку.

        Parameters
        ----------
        update : Update
            Входящее обновление Telegram.
        context : ContextTypes.DEFAULT_TYPE
            Контекст выполнения PTB.

        Returns
        -------
        int
            ``ConversationHandler.END`` после завершения шага.
        """

        if update.message is None or update.message.text is None:
            return ASK_CAPITAL

        text = update.message.text.strip().replace(" ", "").replace(",", ".")
        try:
            capital = float(text)
        except ValueError:
            await update.message.reply_text(
                "Не удалось разобрать число. Введите начальный капитал, например: 100000"
            )
            return ASK_CAPITAL

        if capital <= 0.0:
            await update.message.reply_text("Начальный капитал должен быть больше нуля. Попробуйте снова:")
            return ASK_CAPITAL

        try:
            user_data = _require_user_data(context)
        except RuntimeError:
            await update.message.reply_text("Внутренняя ошибка состояния. Начните снова с /run.")
            return ConversationHandler.END

        start_date = user_data.get("start_date")
        end_date = user_data.get("end_date")
        if not isinstance(start_date, date) or not isinstance(end_date, date):
            await update.message.reply_text("Внутренняя ошибка состояния диалога. Начните снова с /run.")
            return ConversationHandler.END

        chat_id = update.effective_chat.id if update.effective_chat else None
        if chat_id is None:
            return ConversationHandler.END

        await update.message.reply_text("Запускаю расчёт, это может занять некоторое время…")

        try:
            loop = asyncio.get_running_loop()

            def _execute_pipeline() -> PipelineResult:
                return run_pipeline(
                    start_date=_format_iso(start_date),
                    end_date=_format_iso(end_date),
                    initial_capital_rub=capital,
                    security=self._default_security,
                    artifacts_dir=self._artifacts_dir,
                    output_dir=None,
                )

            result = await loop.run_in_executor(None, _execute_pipeline)
        except Exception:
            logger.exception("Ошибка при выполнении production-пайплайна")
            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "Расчёт завершился с ошибкой. Проверьте период, доступность MOEX ISS "
                    "и наличие артефактов моделей."
                ),
            )
            return ConversationHandler.END

        cautious = result.evaluation.strategies.get("cautious")
        greedy = result.evaluation.strategies.get("greedy")
        if cautious is None or greedy is None:
            await context.bot.send_message(
                chat_id=chat_id,
                text="Расчёт выполнен, но отсутствуют ожидаемые стратегии cautious/greedy.",
            )
            return ConversationHandler.END

        await self._send_strategy_chart(
            context=context,
            chat_id=chat_id,
            figure=cautious.chart,
            caption="Стратегия cautious: кривые портфеля по моделям",
        )
        await self._send_strategy_chart(
            context=context,
            chat_id=chat_id,
            figure=greedy.chart,
            caption="Стратегия greedy: кривые портфеля по моделям",
        )

        profit_text = self._format_profit_summary(result.evaluation.summary_json)
        await context.bot.send_message(chat_id=chat_id, text=profit_text)

        try:
            user_data = _require_user_data(context)
        except RuntimeError:
            return ConversationHandler.END

        user_data.pop("start_date", None)
        user_data.pop("end_date", None)
        return ConversationHandler.END

    async def _send_strategy_chart(
        self,
        *,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int,
        figure: Figure,
        caption: str,
    ) -> None:
        """Сохраняет matplotlib Figure в PNG и отправляет как фото в чат.

        Parameters
        ----------
        context : ContextTypes.DEFAULT_TYPE
            Контекст выполнения PTB.
        chat_id : int
            Идентификатор чата для отправки.
        figure : matplotlib.figure.Figure
            Объект ``matplotlib.figure.Figure``.
        caption : str
            Подпись к изображению.
        """

        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
        buffer.seek(0)
        plt.close(figure)
        await context.bot.send_photo(chat_id=chat_id, photo=buffer, caption=caption)

    def _format_profit_summary(self, summary_json: dict[str, dict[str, float]]) -> str:
        """Формирует текст со сводкой прибыли по стратегиям и моделям.

        Parameters
        ----------
        summary_json : dict[str, dict[str, float]]
            Структура из ``EvaluationResult.summary_json``.

        Returns
        -------
        str
            Текст сообщения для пользователя.
        """

        lines: list[str] = ["Сводка прибыли (руб., по моделям):"]
        for strategy_key, profits in summary_json.items():
            lines.append(f"\nСтратегия «{strategy_key}»:")
            if not profits:
                lines.append("  (нет данных)")
                continue
            best_model: str | None = None
            best_profit: float | None = None
            for model_name, profit in profits.items():
                lines.append(f"  • {model_name}: {profit:,.2f} ₽".replace(",", " "))
                if best_profit is None or float(profit) > float(best_profit):
                    best_profit = float(profit)
                    best_model = str(model_name)
            if best_model is not None and best_profit is not None:
                lines.append(
                    f"  Лучшая модель по прибыли: {best_model} ({best_profit:,.2f} ₽)".replace(",", " ")
                )

        return "\n".join(lines)

    async def _cmd_cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Отменяет диалог ``/run`` и очищает сохранённые поля.

        Parameters
        ----------
        update : Update
            Входящее обновление Telegram.
        context : ContextTypes.DEFAULT_TYPE
            Контекст выполнения PTB.

        Returns
        -------
        int
            ``ConversationHandler.END``.
        """

        try:
            user_data = _require_user_data(context)
            user_data.pop("start_date", None)
            user_data.pop("end_date", None)
        except RuntimeError:
            pass
        if update.message:
            await update.message.reply_text("Диалог отменён.")
        return ConversationHandler.END


def trading_bot_from_env() -> TradingTelegramBot:
    """Создаёт ``TradingTelegramBot`` из переменных окружения.

    Ожидаемые переменные:
    - ``TELEGRAM_BOT_TOKEN`` — обязательный токен бота;
    - ``ALLOWED_TELEGRAM_USER_IDS`` — необязательный список ``user_id`` через запятую;
    - ``DEFAULT_SECURITY`` — необязательный тикер (по умолчанию ``SBER``);
    - ``ARTIFACTS_DIR`` — необязательный путь к артефактам (по умолчанию ``models``).

    Returns
    -------
    TradingTelegramBot
        Настроенный экземпляр бота.

    Raises
    ------
    ValueError
        Если отсутствует ``TELEGRAM_BOT_TOKEN``.
    """

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        msg = "Переменная окружения TELEGRAM_BOT_TOKEN не задана или пуста."
        raise ValueError(msg)

    validator = UserValidator.from_env_value(os.environ.get("ALLOWED_TELEGRAM_USER_IDS"))
    security = os.environ.get("DEFAULT_SECURITY", "SBER").strip() or "SBER"
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", "models").strip() or "models"
    return TradingTelegramBot(
        token=token,
        validator=validator,
        default_security=security,
        artifacts_dir=artifacts_dir,
    )


__all__ = [
    "ASK_CAPITAL",
    "ASK_END_DATE",
    "ASK_START_DATE",
    "TradingTelegramBot",
    "UserValidator",
    "WebhookServerConfig",
    "parse_user_date",
    "parse_webhook_config_from_env",
    "trading_bot_from_env",
]

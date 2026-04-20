"""Утилиты для загрузки рыночных данных MOEX в production-пайплайне."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REQUIRED_MARKET_COLUMNS = ("TRADEDATE", "OPEN", "LOW", "HIGH", "LEGALCLOSEPRICE")


@dataclass(frozen=True, slots=True)
class MoexIssConfig:
    """Конфигурация запросов исторических данных к MOEX ISS.

    Parameters
    ----------
    engine : str, default="stock"
        Имя движка ISS.
    market : str, default="shares"
        Имя рынка ISS.
    board : str, default="TQBR"
        Идентификатор торгового режима.
    timeout_seconds : float, default=20.0
        HTTP-таймаут для каждого запроса.
    page_size : int, default=100
        Ожидаемый размер страницы на стороне сервера при пагинации.
    base_url : str, default="https://iss.moex.com"
        Базовый URL MOEX ISS.
    """

    engine: str = "stock"
    market: str = "shares"
    board: str = "TQBR"
    timeout_seconds: float = 20.0
    page_size: int = 100
    base_url: str = "https://iss.moex.com"


class MoexIssDataLoader:
    """Загружает OHLC-историю инструмента с MOEX через ISS.

    Parameters
    ----------
    config : MoexIssConfig or None, optional
        Параметры запросов. Если не переданы, используются значения по умолчанию.
    session : requests.Session or None, optional
        Объект сессии для HTTP-запросов. Если не передан, создаётся новая сессия.
    """

    def __init__(
        self,
        config: MoexIssConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        """Инициализирует загрузчик.

        Parameters
        ----------
        config : MoexIssConfig or None, optional
            Параметры запросов.
        session : requests.Session or None, optional
            HTTP-сессия для запросов к ISS.
        """

        self._config = config or MoexIssConfig()
        self._session = session or requests.Session()

    def load_history(
        self,
        security: str,
        start_date: date | str,
        end_date: date | str,
    ) -> pd.DataFrame:
        """Загружает и нормализует дневную историю инструмента.

        Parameters
        ----------
        security : str
            Тикер инструмента, например ``"SBER"``.
        start_date : date or str
            Начало периода включительно.
        end_date : date or str
            Конец периода включительно.

        Returns
        -------
        pd.DataFrame
            Нормализованные рыночные данные с колонками, совместимыми
            с обучающим пайплайном: ``TRADEDATE``, ``OPEN``, ``LOW``,
            ``HIGH`` и ``LEGALCLOSEPRICE``.

        Raises
        ------
        ValueError
            Если период некорректен, ответ имеет неверный формат
            или сервер не вернул ни одной строки.
        RuntimeError
            Если MOEX ISS недоступен или вернул неуспешный HTTP-статус.
        """

        normalized_security = security.strip().upper()
        if not normalized_security:
            msg = "security must be a non-empty string"
            raise ValueError(msg)

        start_value = self._normalize_date(start_date)
        end_value = self._normalize_date(end_date)
        if start_value > end_value:
            msg = "start_date must be less than or equal to end_date"
            raise ValueError(msg)

        payload = self._load_history_payload(
            security=normalized_security,
            start_date=start_value,
            end_date=end_value,
        )
        return self._normalize_history_frame(payload)

    def _load_history_payload(
        self,
        *,
        security: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Загружает все страницы исторических данных из MOEX ISS.

        Parameters
        ----------
        security : str
            Тикер в верхнем регистре.
        start_date : str
            Начало периода включительно в ISO-формате.
        end_date : str
            Конец периода включительно в ISO-формате.

        Returns
        -------
        dict[str, Any]
            Словарь с объединёнными блоками ``columns`` и ``data`` из ISS.
        """

        all_rows: list[list[Any]] = []
        columns: list[str] | None = None
        offset = 0

        while True:
            page = self._fetch_history_page(
                security=security,
                start_date=start_date,
                end_date=end_date,
                start=offset,
            )
            page_columns = page["columns"]
            page_rows = page["data"]
            if columns is None:
                columns = page_columns
            elif columns != page_columns:
                msg = "MOEX ISS returned inconsistent history columns across pages"
                raise ValueError(msg)

            if not page_rows:
                break

            all_rows.extend(page_rows)
            if len(page_rows) < self._config.page_size:
                break
            offset += self._config.page_size

        if columns is None or not all_rows:
            msg = (
                f"MOEX ISS returned no history rows for {security!r} "
                f"between {start_date} and {end_date}"
            )
            raise ValueError(msg)

        return {"columns": columns, "data": all_rows}

    def _fetch_history_page(
        self,
        *,
        security: str,
        start_date: str,
        end_date: str,
        start: int,
    ) -> dict[str, list[Any]]:
        """Загружает одну страницу истории из MOEX ISS.

        Parameters
        ----------
        security : str
            Тикер в верхнем регистре.
        start_date : str
            Начало периода включительно в ISO-формате.
        end_date : str
            Конец периода включительно в ISO-формате.
        start : int
            Смещение для пагинации ISS.

        Returns
        -------
        dict[str, list[Any]]
            Словарь с ``columns`` и ``data`` из блока ``history``.
        """

        url = self._build_history_url(security)
        params = {
            "from": start_date,
            "till": end_date,
            "start": start,
            "iss.meta": "off",
        }
        try:
            response = self._session.get(url, params=params, timeout=self._config.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            msg = f"Failed to load MOEX ISS history for {security!r}: {exc}"
            raise RuntimeError(msg) from exc

        payload = response.json()
        history_block = payload.get("history")
        if not isinstance(history_block, dict):
            msg = "MOEX ISS response does not contain a valid 'history' block"
            raise ValueError(msg)

        columns = history_block.get("columns")
        rows = history_block.get("data")
        if not isinstance(columns, list) or not isinstance(rows, list):
            msg = "MOEX ISS history block must contain 'columns' and 'data' lists"
            raise ValueError(msg)

        return {"columns": columns, "data": rows}

    def _build_history_url(self, security: str) -> str:
        """Собирает URL эндпоинта истории MOEX ISS.

        Parameters
        ----------
        security : str
            Тикер в верхнем регистре.

        Returns
        -------
        str
            Полный URL эндпоинта ISS.
        """

        base_path = Path("iss") / "history" / "engines" / self._config.engine / "markets" / self._config.market
        board_path = base_path / "boards" / self._config.board / "securities" / f"{security}.json"
        return f"{self._config.base_url.rstrip('/')}/{board_path.as_posix()}"

    def _normalize_history_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Преобразует сырой ответ ISS во внутреннюю схему рыночных данных.

        Parameters
        ----------
        payload : dict[str, Any]
            Сырой объединённый словарь с ключами ``columns`` и ``data``.

        Returns
        -------
        pd.DataFrame
            Отсортированные и провалидированные дневные рыночные данные.
        """

        frame = pd.DataFrame(payload["data"], columns=payload["columns"])
        missing_columns = [column for column in REQUIRED_MARKET_COLUMNS if column not in frame.columns]
        if missing_columns:
            msg = f"MOEX ISS response is missing required columns: {missing_columns}"
            raise ValueError(msg)

        normalized = frame.loc[:, list(REQUIRED_MARKET_COLUMNS)].copy()
        normalized["TRADEDATE"] = pd.to_datetime(normalized["TRADEDATE"], errors="coerce").dt.strftime("%Y-%m-%d")
        if normalized["TRADEDATE"].isna().any():
            msg = "MOEX ISS response contains invalid TRADEDATE values"
            raise ValueError(msg)

        for column in REQUIRED_MARKET_COLUMNS[1:]:
            numeric_values = pd.to_numeric(normalized[column], errors="coerce")
            if numeric_values.isna().any():
                msg = f"MOEX ISS response contains invalid numeric values in {column!r}"
                raise ValueError(msg)
            normalized[column] = numeric_values.astype(float)

        normalized = normalized.sort_values("TRADEDATE").drop_duplicates(subset=["TRADEDATE"], keep="last")
        normalized = normalized.reset_index(drop=True)
        if normalized.empty:
            msg = "Normalized MOEX ISS history is empty"
            raise ValueError(msg)
        return normalized

    def _normalize_date(self, value: date | str) -> str:
        """Преобразует значение, похожее на дату, в ISO-формат.

        Parameters
        ----------
        value : date or str
            Входное значение для нормализации.

        Returns
        -------
        str
            Дата в формате ``YYYY-MM-DD``.
        """

        if isinstance(value, date):
            return value.isoformat()
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            msg = f"Invalid date value: {value!r}"
            raise ValueError(msg)
        return timestamp.date().isoformat()


__all__ = ["MoexIssConfig", "MoexIssDataLoader", "REQUIRED_MARKET_COLUMNS"]

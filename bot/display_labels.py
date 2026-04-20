"""Человекочитаемые подписи стратегий и моделей для UI (внутренние ключи не меняются)."""

from __future__ import annotations

STRATEGY_DISPLAY_NAMES: dict[str, str] = {
    "cautious": "Осторожная",
    "greedy": "Жадная",
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "linear_regression": "Линейная регрессия",
    "logistic_regression": "Логистическая регрессия",
    "catboost_classifier": "Градиентный бустинг",
    "buy_and_hold": "Купить и удерживать",
}


def strategy_display_name(key: str) -> str:
    """Возвращает русскоязычную подпись для ключа стратегии.

    Если ключа нет в словаре, возвращается исходная строка.

    Parameters
    ----------
    key : str
        Внутренний идентификатор стратегии (например, ``\"cautious\"``).

    Returns
    -------
    str
        Строка для отображения пользователю.
    """

    return STRATEGY_DISPLAY_NAMES.get(key, key)


def model_display_name(key: str) -> str:
    """Возвращает русскоязычную подпись для ключа модели.

    Если ключа нет в словаре, возвращается исходная строка.

    Parameters
    ----------
    key : str
        Внутренний идентификатор модели (например, ``\"linear_regression\"``).

    Returns
    -------
    str
        Строка для отображения пользователю.
    """

    return MODEL_DISPLAY_NAMES.get(key, key)


__all__ = [
    "MODEL_DISPLAY_NAMES",
    "STRATEGY_DISPLAY_NAMES",
    "model_display_name",
    "strategy_display_name",
]

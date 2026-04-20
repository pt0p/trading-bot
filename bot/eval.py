"""Инференс моделей, оценка стратегий и построение графиков."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any

PROD_CACHE_DIR = Path(tempfile.gettempdir()) / "trading_bot_prod_cache"
PROD_MPL_CACHE_DIR = PROD_CACHE_DIR / "mplconfig"
PROD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROD_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(PROD_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROD_MPL_CACHE_DIR))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load as joblib_load
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, NullLocator

from bot.feature_extractor import MODEL_ORDER, ModelName, PreparedModelDataset

STRATEGY_NAMES = ("cautious", "greedy")


@dataclass(frozen=True, slots=True)
class StrategyEvaluationResult:
    """Результаты для одного семейства стратегий.

    Parameters
    ----------
    summary : pd.DataFrame
        Сводка по моделям с итоговой стоимостью портфеля, прибылью и доходностью.
    curves : pd.DataFrame
        Кривые портфеля для всех сравниваемых моделей.
    chart : Figure
        Объект Matplotlib Figure для семейства стратегий.
    """

    summary: pd.DataFrame
    curves: pd.DataFrame
    chart: Figure


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Объединённый результат по всем стратегиям.

    Parameters
    ----------
    summary_json : dict[str, dict[str, float]]
        Вложенный словарь, готовый к сериализации в JSON, с прибылью по стратегиям и моделям.
    strategies : dict[str, StrategyEvaluationResult]
        Полные результаты по каждой стратегии.
    """

    summary_json: dict[str, dict[str, float]]
    strategies: dict[str, StrategyEvaluationResult]


class StrategyEvaluator:
    """Выполняет инференс и бэктесты для обученных артефактов."""

    def evaluate(
        self,
        datasets: dict[ModelName, PreparedModelDataset],
        initial_capital_rub: float,
    ) -> EvaluationResult:
        """Оценивает все модели в осторожной и жадной стратегиях.

        Parameters
        ----------
        datasets : dict[ModelName, PreparedModelDataset]
            Готовые к инференсу датасеты, сгруппированные по имени модели.
        initial_capital_rub : float
            Начальный баланс портфеля.

        Returns
        -------
        EvaluationResult
            Итоговая JSON-готовая сводка, кривые и графики.
        """

        self._validate_initial_capital(initial_capital_rub)
        aligned = self._align_datasets(datasets)

        cautious_curves = self._evaluate_cautious(aligned, initial_capital_rub)
        greedy_curves = self._evaluate_greedy(aligned, initial_capital_rub)

        cautious_result = self._build_strategy_result(
            strategy_name="cautious",
            curves_by_model=cautious_curves,
            initial_capital_rub=initial_capital_rub,
        )
        greedy_result = self._build_strategy_result(
            strategy_name="greedy",
            curves_by_model=greedy_curves,
            initial_capital_rub=initial_capital_rub,
        )

        summary_json = {
            "cautious": self._summary_to_profit_map(cautious_result.summary),
            "greedy": self._summary_to_profit_map(greedy_result.summary),
        }
        return EvaluationResult(
            summary_json=summary_json,
            strategies={
                "cautious": cautious_result,
                "greedy": greedy_result,
            },
        )

    def _align_datasets(
        self,
        datasets: dict[ModelName, PreparedModelDataset],
    ) -> dict[ModelName, PreparedModelDataset]:
        """Выравнивает датасеты всех моделей по общему пересечению дат.

        Parameters
        ----------
        datasets : dict[ModelName, PreparedModelDataset]
            Подготовленные датасеты до выравнивания.

        Returns
        -------
        dict[ModelName, PreparedModelDataset]
            Датасеты, выровненные по общему набору дат.
        """

        common_times: set[str] | None = None
        for model_name in MODEL_ORDER:
            model_dataset = datasets[model_name].dataset
            times = set(model_dataset["time"].astype(str).tolist())
            common_times = times if common_times is None else common_times.intersection(times)

        if not common_times:
            msg = "Prepared model datasets do not share a common backtest period"
            raise ValueError(msg)

        ordered_times = sorted(common_times)
        aligned: dict[ModelName, PreparedModelDataset] = {}
        for model_name in MODEL_ORDER:
            prepared = datasets[model_name]
            filtered = prepared.dataset[prepared.dataset["time"].astype(str).isin(ordered_times)].copy()
            filtered["time"] = filtered["time"].astype(str)
            filtered = filtered.sort_values("time").reset_index(drop=True)
            if filtered.empty:
                msg = f"Aligned dataset for {model_name!r} is empty"
                raise ValueError(msg)
            aligned[model_name] = PreparedModelDataset(artifact=prepared.artifact, dataset=filtered)
        return aligned

    def _evaluate_cautious(
        self,
        datasets: dict[ModelName, PreparedModelDataset],
        initial_capital_rub: float,
    ) -> dict[str, pd.DataFrame]:
        """Оценивает семейство стратегий с пропорциональной аллокацией.

        Parameters
        ----------
        datasets : dict[ModelName, PreparedModelDataset]
            Датасеты, выровненные по датам.
        initial_capital_rub : float
            Начальный баланс портфеля.

        Returns
        -------
        dict[str, pd.DataFrame]
            Кривые, сгруппированные по имени модели.
        """

        curves: dict[str, pd.DataFrame] = {}
        reference_time = datasets["linear_regression"].dataset["time"]
        reference_returns = datasets["linear_regression"].dataset["return_t"].to_numpy(dtype=float, copy=False)

        for model_name in MODEL_ORDER:
            prepared = datasets[model_name]
            model = self._load_model(prepared.artifact.model_path)
            if model_name == "linear_regression":
                raw_predictions = self._predict_numeric(model, prepared.feature_frame, model_path=prepared.artifact.model_path)
                probabilities = self._sigmoid(raw_predictions)
                curve = self._simulate_proportional(
                    returns=prepared.dataset["return_t"].to_numpy(dtype=float, copy=False),
                    probabilities=probabilities,
                    initial_capital_rub=initial_capital_rub,
                    neutral_probability=0.5,
                )
            else:
                probabilities = self._predict_positive_proba(
                    model,
                    prepared.feature_frame,
                    model_path=prepared.artifact.model_path,
                )
                neutral_probability = prepared.artifact.decision_threshold or 0.5
                curve = self._simulate_proportional(
                    returns=prepared.dataset["return_t"].to_numpy(dtype=float, copy=False),
                    probabilities=probabilities,
                    initial_capital_rub=initial_capital_rub,
                    neutral_probability=neutral_probability,
                )

            curves[model_name] = self._attach_curve_metadata(curve, reference_time, model_name)

        buy_and_hold_curve = self._simulate_buy_and_hold(
            returns=reference_returns,
            initial_capital_rub=initial_capital_rub,
        )
        curves["buy_and_hold"] = self._attach_curve_metadata(buy_and_hold_curve, reference_time, "buy_and_hold")
        return curves

    def _evaluate_greedy(
        self,
        datasets: dict[ModelName, PreparedModelDataset],
        initial_capital_rub: float,
    ) -> dict[str, pd.DataFrame]:
        """Оценивает семейство стратегий с полным входом в позицию.

        Parameters
        ----------
        datasets : dict[ModelName, PreparedModelDataset]
            Датасеты, выровненные по датам.
        initial_capital_rub : float
            Начальный баланс портфеля.

        Returns
        -------
        dict[str, pd.DataFrame]
            Кривые, сгруппированные по имени модели.
        """

        curves: dict[str, pd.DataFrame] = {}
        reference_time = datasets["linear_regression"].dataset["time"]
        reference_returns = datasets["linear_regression"].dataset["return_t"].to_numpy(dtype=float, copy=False)

        for model_name in MODEL_ORDER:
            prepared = datasets[model_name]
            model = self._load_model(prepared.artifact.model_path)
            if model_name == "linear_regression":
                raw_predictions = self._predict_numeric(model, prepared.feature_frame, model_path=prepared.artifact.model_path)
                bullish = raw_predictions >= 0.0
            else:
                probabilities = self._predict_positive_proba(
                    model,
                    prepared.feature_frame,
                    model_path=prepared.artifact.model_path,
                )
                threshold = prepared.artifact.decision_threshold or 0.5
                bullish = probabilities >= threshold

            curve = self._simulate_full_position(
                returns=prepared.dataset["return_t"].to_numpy(dtype=float, copy=False),
                bullish=bullish,
                initial_capital_rub=initial_capital_rub,
            )
            curves[model_name] = self._attach_curve_metadata(curve, reference_time, model_name)

        buy_and_hold_curve = self._simulate_buy_and_hold(
            returns=reference_returns,
            initial_capital_rub=initial_capital_rub,
        )
        curves["buy_and_hold"] = self._attach_curve_metadata(buy_and_hold_curve, reference_time, "buy_and_hold")
        return curves

    def _build_strategy_result(
        self,
        *,
        strategy_name: str,
        curves_by_model: dict[str, pd.DataFrame],
        initial_capital_rub: float,
    ) -> StrategyEvaluationResult:
        """Собирает сводку и график для одного семейства стратегий.

        Parameters
        ----------
        strategy_name : str
            Имя семейства стратегий.
        curves_by_model : dict[str, pd.DataFrame]
            Кривые портфеля для всех сравниваемых моделей.
        initial_capital_rub : float
            Начальный баланс портфеля.

        Returns
        -------
        StrategyEvaluationResult
            Готовый объект результата стратегии.
        """

        curves = pd.concat(curves_by_model.values(), ignore_index=True)
        summary_rows = [
            self._build_summary_row(curve_df, initial_capital_rub=initial_capital_rub)
            for curve_df in curves_by_model.values()
        ]
        summary = pd.DataFrame(summary_rows)
        chart = self._build_portfolio_chart(curves, strategy_name=strategy_name, initial_capital_rub=initial_capital_rub)
        return StrategyEvaluationResult(summary=summary, curves=curves, chart=chart)

    def _summary_to_profit_map(self, summary: pd.DataFrame) -> dict[str, float]:
        """Преобразует сводную таблицу в JSON-готовую карту прибыли.

        Parameters
        ----------
        summary : pd.DataFrame
            Сводная таблица по стратегии.

        Returns
        -------
        dict[str, float]
            Прибыль по имени модели.
        """

        return {
            str(row["model"]): round(float(row["profit_rub"]), 6)
            for _, row in summary.iterrows()
        }

    def _load_model(self, path: Path) -> object:
        """Загружает сериализованную модель с диска.

        Parameters
        ----------
        path : Path
            Путь к сериализованной модели.

        Returns
        -------
        object
            Экземпляр обученной модели.
        """

        suffix = path.suffix.lower()
        try:
            if suffix == ".joblib":
                return joblib_load(path)
            if suffix == ".cbm":
                from catboost import CatBoostClassifier

                model = CatBoostClassifier(verbose=False)
                model.load_model(str(path))
                return model
        except Exception as exc:
            msg = f"Failed to load model from {path}: {exc}"
            raise ValueError(msg) from exc

        msg = f"Unsupported model extension for {path}"
        raise ValueError(msg)

    def _predict_numeric(self, model: object, features: pd.DataFrame, *, model_path: Path) -> np.ndarray:
        """Вызывает ``predict`` и приводит результат к числовому вектору.

        Parameters
        ----------
        model : object
            Обученный estimator.
        features : pd.DataFrame
            DataFrame с фичами в корректном порядке.
        model_path : Path
            Путь к артефакту для сообщений об ошибках.

        Returns
        -------
        np.ndarray
            Одномерный вектор предсказаний.
        """

        predict = getattr(model, "predict", None)
        if predict is None:
            msg = f"Model {model_path} does not implement predict"
            raise ValueError(msg)
        try:
            predictions = np.asarray(predict(features), dtype=float).reshape(-1)
        except Exception as exc:
            msg = f"Failed to generate predictions with {model_path}: {exc}"
            raise ValueError(msg) from exc
        if len(predictions) != len(features):
            msg = f"Prediction count mismatch for {model_path}"
            raise ValueError(msg)
        return predictions

    def _predict_positive_proba(
        self,
        model: object,
        features: pd.DataFrame,
        *,
        model_path: Path,
    ) -> np.ndarray:
        """Вызывает ``predict_proba`` и извлекает вероятность положительного класса.

        Parameters
        ----------
        model : object
            Обученный estimator.
        features : pd.DataFrame
            DataFrame с фичами в корректном порядке.
        model_path : Path
            Путь к артефакту для сообщений об ошибках.

        Returns
        -------
        np.ndarray
            Вероятности положительного класса.
        """

        predict_proba = getattr(model, "predict_proba", None)
        if predict_proba is None:
            msg = f"Model {model_path} does not implement predict_proba"
            raise ValueError(msg)
        try:
            raw_scores = np.asarray(predict_proba(features), dtype=float)
        except Exception as exc:
            msg = f"Failed to generate predict_proba output for {model_path}: {exc}"
            raise ValueError(msg) from exc
        if raw_scores.ndim != 2 or raw_scores.shape[1] < 2:
            msg = f"predict_proba for {model_path} must return shape (n_samples, n_classes)"
            raise ValueError(msg)
        probabilities = raw_scores[:, 1].reshape(-1)
        if len(probabilities) != len(features):
            msg = f"Probability count mismatch for {model_path}"
            raise ValueError(msg)
        return probabilities

    def _simulate_proportional(
        self,
        *,
        returns: np.ndarray,
        probabilities: np.ndarray,
        initial_capital_rub: float,
        neutral_probability: float,
    ) -> pd.DataFrame:
        """Симулирует стратегию с пропорциональной аллокацией.

        Parameters
        ----------
        returns : np.ndarray
            Реализованные доходности следующего шага.
        probabilities : np.ndarray
            Вероятности роста в диапазоне ``[0, 1]``.
        initial_capital_rub : float
            Начальный баланс портфеля.
        neutral_probability : float
            Вероятность, при которой целевая экспозиция в акции становится нейтральной.

        Returns
        -------
        pd.DataFrame
            Состояние портфеля на каждом шаге.
        """

        returns_array = self._coerce_numeric_array(returns, field_name="returns")
        probability_array = self._coerce_numeric_array(probabilities, field_name="probabilities")
        if len(returns_array) != len(probability_array):
            msg = "returns and probabilities lengths must match"
            raise ValueError(msg)
        alphas = self._probabilities_to_alpha(probability_array, neutral_probability=neutral_probability)
        return self._simulate_from_alpha(
            returns=returns_array,
            alpha=alphas,
            initial_capital_rub=initial_capital_rub,
        )

    def _simulate_full_position(
        self,
        *,
        returns: np.ndarray,
        bullish: np.ndarray,
        initial_capital_rub: float,
    ) -> pd.DataFrame:
        """Симулирует бинарную стратегию с полным входом в позицию.

        Parameters
        ----------
        returns : np.ndarray
            Реализованные доходности следующего шага.
        bullish : np.ndarray
            Булев вектор сигналов на покупку.
        initial_capital_rub : float
            Начальный баланс портфеля.

        Returns
        -------
        pd.DataFrame
            Состояние портфеля на каждом шаге.
        """

        returns_array = self._coerce_numeric_array(returns, field_name="returns")
        bullish_array = np.asarray(bullish, dtype=bool).reshape(-1)
        if len(returns_array) != len(bullish_array):
            msg = "returns and bullish lengths must match"
            raise ValueError(msg)

        cash_rub = float(initial_capital_rub)
        stock_value_rub = 0.0
        rows: list[dict[str, float]] = []
        for is_bullish, return_t in zip(bullish_array, returns_array, strict=True):
            total_rub = cash_rub + stock_value_rub
            if is_bullish:
                cash_rub = 0.0
                stock_value_rub = total_rub
            else:
                cash_rub = total_rub
                stock_value_rub = 0.0

            stock_value_rub *= 1.0 + return_t
            rows.append(
                {
                    "cash_rub": cash_rub,
                    "stock_value_rub": stock_value_rub,
                    "portfolio_value_rub": cash_rub + stock_value_rub,
                }
            )
        return pd.DataFrame(rows)

    def _simulate_buy_and_hold(
        self,
        *,
        returns: np.ndarray,
        initial_capital_rub: float,
    ) -> pd.DataFrame:
        """Симулирует бенчмарк buy-and-hold.

        Parameters
        ----------
        returns : np.ndarray
            Реализованные доходности следующего шага.
        initial_capital_rub : float
            Начальный баланс портфеля.

        Returns
        -------
        pd.DataFrame
            Состояние портфеля на каждом шаге.
        """

        returns_array = self._coerce_numeric_array(returns, field_name="returns")
        stock_value_rub = float(initial_capital_rub)
        rows: list[dict[str, float]] = []
        for return_t in returns_array:
            stock_value_rub *= 1.0 + return_t
            rows.append(
                {
                    "cash_rub": 0.0,
                    "stock_value_rub": stock_value_rub,
                    "portfolio_value_rub": stock_value_rub,
                }
            )
        return pd.DataFrame(rows)

    def _simulate_from_alpha(
        self,
        *,
        returns: np.ndarray,
        alpha: np.ndarray,
        initial_capital_rub: float,
    ) -> pd.DataFrame:
        """Симулирует динамический портфель по целевым значениям alpha.

        Parameters
        ----------
        returns : np.ndarray
            Реализованные доходности следующего шага.
        alpha : np.ndarray
            Желаемая экспозиция портфеля в диапазоне ``[-1, 1]``.
        initial_capital_rub : float
            Начальный баланс портфеля.

        Returns
        -------
        pd.DataFrame
            Состояние портфеля на каждом шаге.
        """

        returns_array = self._coerce_numeric_array(returns, field_name="returns")
        alpha_array = self._coerce_numeric_array(alpha, field_name="alpha")
        if len(returns_array) != len(alpha_array):
            msg = "returns and alpha lengths must match"
            raise ValueError(msg)

        cash_rub = float(initial_capital_rub)
        stock_value_rub = 0.0
        rows: list[dict[str, float]] = []
        for alpha_t, return_t in zip(alpha_array, returns_array, strict=True):
            clipped_alpha = float(np.clip(alpha_t, -1.0, 1.0))
            if clipped_alpha > 0.0:
                buy_rub = clipped_alpha * cash_rub
                cash_rub -= buy_rub
                stock_value_rub += buy_rub
            elif clipped_alpha < 0.0:
                sell_rub = abs(clipped_alpha) * stock_value_rub
                stock_value_rub -= sell_rub
                cash_rub += sell_rub

            stock_value_rub *= 1.0 + return_t
            rows.append(
                {
                    "cash_rub": cash_rub,
                    "stock_value_rub": stock_value_rub,
                    "portfolio_value_rub": cash_rub + stock_value_rub,
                }
            )
        return pd.DataFrame(rows)

    def _probabilities_to_alpha(self, probabilities: np.ndarray, *, neutral_probability: float) -> np.ndarray:
        """Преобразует вероятности роста в целевые значения экспозиции.

        Parameters
        ----------
        probabilities : np.ndarray
            Вероятности роста.
        neutral_probability : float
            Вероятность, соответствующая нулевой экспозиции в акции.

        Returns
        -------
        np.ndarray
            Вектор экспозиции в диапазоне ``[-1, 1]``.
        """

        tau = float(neutral_probability)
        if not 0.0 < tau < 1.0:
            msg = "neutral_probability must be strictly between 0 and 1"
            raise ValueError(msg)
        left = (probabilities - tau) / tau
        right = (probabilities - tau) / (1.0 - tau)
        return np.clip(np.where(probabilities <= tau, left, right), -1.0, 1.0)

    def _attach_curve_metadata(
        self,
        curve_df: pd.DataFrame,
        time: pd.Series,
        model_name: str,
    ) -> pd.DataFrame:
        """Добавляет даты и метки к симулированной кривой.

        Parameters
        ----------
        curve_df : pd.DataFrame
            Симулированное состояние портфеля.
        time : pd.Series
            Упорядоченный ряд дат.
        model_name : str
            Метка кривой.

        Returns
        -------
        pd.DataFrame
            DataFrame кривой с метаданными.
        """

        if len(curve_df) != len(time):
            msg = f"Curve length for {model_name!r} does not match the time index"
            raise ValueError(msg)
        return pd.DataFrame(
            {
                "time": pd.to_datetime(time, errors="coerce").to_numpy(copy=False),
                "model": model_name,
                "cash_rub": curve_df["cash_rub"].to_numpy(copy=False),
                "stock_value_rub": curve_df["stock_value_rub"].to_numpy(copy=False),
                "portfolio_value_rub": curve_df["portfolio_value_rub"].to_numpy(copy=False),
            }
        )

    def _build_summary_row(self, curve_df: pd.DataFrame, *, initial_capital_rub: float) -> dict[str, Any]:
        """Строит одну строку сводки по размеченной кривой.

        Parameters
        ----------
        curve_df : pd.DataFrame
            Размеченная кривая портфеля.
        initial_capital_rub : float
            Начальный баланс портфеля.

        Returns
        -------
        dict[str, Any]
            Одна строка сводки.
        """

        last_row = curve_df.iloc[-1]
        final_value = float(last_row["portfolio_value_rub"])
        profit = final_value - float(initial_capital_rub)
        return {
            "model": str(last_row["model"]),
            "final_portfolio_value_rub": final_value,
            "profit_rub": profit,
            "return_fraction": final_value / float(initial_capital_rub) - 1.0,
        }

    def _build_portfolio_chart(
        self,
        curves: pd.DataFrame,
        *,
        strategy_name: str,
        initial_capital_rub: float,
    ) -> Figure:
        """Строит линейный график для одного семейства стратегий.

        Parameters
        ----------
        curves : pd.DataFrame
            Кривые портфеля для всех моделей внутри семейства стратегий.
        strategy_name : str
            Имя семейства стратегий.
        initial_capital_rub : float
            Начальный баланс портфеля.

        Returns
        -------
        Figure
            Объект Matplotlib Figure.
        """

        figure, axis = plt.subplots(figsize=(12, 6.5))
        for model_name, model_curve in curves.groupby("model", sort=False):
            ordered = model_curve.sort_values("time")
            axis.plot(
                ordered["time"].to_numpy(copy=False),
                ordered["portfolio_value_rub"].to_numpy(copy=False),
                label=model_name,
            )

        axis.axhline(
            y=float(initial_capital_rub),
            color="0.55",
            linestyle="--",
            linewidth=1.2,
            zorder=0,
        )
        axis.set_title(f"{strategy_name.capitalize()} strategy portfolio curves")
        axis.set_xlabel("Дата")
        axis.set_ylabel("Баланс портфеля, RUB")
        axis.yaxis.set_major_formatter(FuncFormatter(self._format_balance_thousands))
        axis.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.tick_params(axis="x", rotation=25, labelbottom=True)
        axis.legend()
        figure.tight_layout(rect=(0.03, 0.12, 0.97, 0.98))
        return figure

    def _format_balance_thousands(self, value: float, _pos: int | None) -> str:
        """Форматирует значения оси Y как целочисленные балансы с группировкой.

        Parameters
        ----------
        value : float
            Числовое значение отметки.
        _pos : int or None
            Аргумент callback-функции Matplotlib.

        Returns
        -------
        str
            Отформатированная подпись отметки.
        """

        if not np.isfinite(value):
            return ""
        return f"{int(round(value)):,}".replace(",", " ")

    def _coerce_numeric_array(self, values: np.ndarray | pd.Series, *, field_name: str) -> np.ndarray:
        """Проверяет, что вход можно трактовать как корректный числовой вектор.

        Parameters
        ----------
        values : np.ndarray or pd.Series
            Входной вектор.
        field_name : str
            Имя поля для сообщений об ошибках.

        Returns
        -------
        np.ndarray
            Одномерный числовой вектор.
        """

        array = np.asarray(values, dtype=float).reshape(-1)
        if np.isnan(array).any():
            msg = f"{field_name} contains NaN values"
            raise ValueError(msg)
        return array

    def _sigmoid(self, values: np.ndarray) -> np.ndarray:
        """Применяет численно устойчивое сигмоидное преобразование.

        Parameters
        ----------
        values : np.ndarray
            Входные значения score.

        Returns
        -------
        np.ndarray
            Вероятности в диапазоне ``[0, 1]``.
        """

        clipped_values = np.clip(values, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-clipped_values))

    def _validate_initial_capital(self, initial_capital_rub: float) -> None:
        """Проверяет корректность начального баланса портфеля.

        Parameters
        ----------
        initial_capital_rub : float
            Начальный баланс портфеля.
        """

        if float(initial_capital_rub) <= 0.0:
            msg = "initial_capital_rub must be > 0"
            raise ValueError(msg)


__all__ = ["EvaluationResult", "STRATEGY_NAMES", "StrategyEvaluationResult", "StrategyEvaluator"]

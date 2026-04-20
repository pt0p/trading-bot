"""Публичный orchestration API для production-торгового пайплайна."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from bot.data_loader import MoexIssConfig, MoexIssDataLoader
from bot.eval import EvaluationResult, StrategyEvaluationResult, StrategyEvaluator
from bot.feature_extractor import ArtifactRegistry, FeatureExtractor, ModelName, PreparedModelDataset

MOEX_HISTORY_LOOKBACK_MONTHS = 1


def _to_iso_date(value: str) -> str:
    """Преобразует дату в строку ``YYYY-MM-DD`` для сравнения с полями датасетов.

    Parameters
    ----------
    value : str
        Дата в формате, совместимом с :class:`pandas.Timestamp`.

    Returns
    -------
    str
        Календарная дата в ISO-8601.
    """

    ts = pd.Timestamp(value)
    if pd.isna(ts):
        msg = f"Invalid date value: {value!r}"
        raise ValueError(msg)
    return ts.date().isoformat()


def _moex_fetch_start_date(user_start_date: str, *, lookback_months: int) -> str:
    """Сдвигает дату начала запроса к MOEX назад на ``lookback_months`` для прогрева лагов.

    Parameters
    ----------
    user_start_date : str
        Нижняя граница периода, заданного пользователем (включительно).
    lookback_months : int
        Число календарных месяцев, на которое расширяется окно загрузки.

    Returns
    -------
    str
        Дата ``from`` для запроса ISS в ISO-8601 (не позже необходимого минимума биржевой истории).

    Raises
    ------
    ValueError
        Если ``user_start_date`` не удаётся распарсить.

    Notes
    -----
    Рыночные данные за расширенное окно используются только для корректных лагов и rolling.
    Отчётная зона пользователя задаётся отдельным обрезанием после построения фичей.

    """

    anchor = pd.Timestamp(user_start_date)
    if pd.isna(anchor):
        msg = f"Invalid date value: {user_start_date!r}"
        raise ValueError(msg)
    expanded = anchor - pd.DateOffset(months=lookback_months)
    return expanded.date().isoformat()


def _trim_market_data_to_window(
    market_data: pd.DataFrame,
    *,
    start_iso: str,
    end_iso: str,
) -> pd.DataFrame:
    """Оставляет строки ``TRADEDATE`` внутри ``[start_iso, end_iso]``.

    Parameters
    ----------
    market_data : pd.DataFrame
        Нормализованный фрейм загрузчика MOEX ISS.
    start_iso : str
        Начало включительно, ``YYYY-MM-DD``.
    end_iso : str
        Конец включительно, ``YYYY-MM-DD``.

    Returns
    -------
    pd.DataFrame
        Отфильтрованный и переиндексированный фрейм.

    Raises
    ------
    ValueError
        Если после фильтрации данных не осталось.

    """

    dates = market_data["TRADEDATE"].astype(str)
    mask = (dates >= start_iso) & (dates <= end_iso)
    trimmed = market_data.loc[mask].reset_index(drop=True)
    if trimmed.empty:
        msg = f"No market rows in requested window [{start_iso}, {end_iso}]"
        raise ValueError(msg)
    return trimmed


def _trim_prepared_datasets_to_window(
    datasets: dict[ModelName, PreparedModelDataset],
    *,
    start_iso: str,
    end_iso: str,
) -> dict[ModelName, PreparedModelDataset]:
    """Обрезает подготовленные датасеты по календарному окну пользователя.

    Parameters
    ----------
    datasets : dict[ModelName, PreparedModelDataset]
        Датасеты после полного построения фичей на расширенной истории.
    start_iso : str
        Начало включительно.
    end_iso : str
        Конец включительно.

    Returns
    -------
    dict[ModelName, PreparedModelDataset]
        Новые обёртки с урезанными ``dataset``.

    Raises
    ------
    ValueError
        Если для какой-либо модели в окне не осталось строк.

    """

    trimmed: dict[ModelName, PreparedModelDataset] = {}
    for model_name, prepared in datasets.items():
        frame = prepared.dataset.copy()
        times = frame["time"].astype(str)
        mask = (times >= start_iso) & (times <= end_iso)
        filtered = frame.loc[mask].sort_values("time").reset_index(drop=True)
        if filtered.empty:
            msg = f"No feature rows in requested window [{start_iso}, {end_iso}] " f"for model {model_name!r}"
            raise ValueError(msg)
        trimmed[model_name] = PreparedModelDataset(artifact=prepared.artifact, dataset=filtered)
    return trimmed


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Конфигурация production-пайплайна.

    Parameters
    ----------
    start_date : str
        Начало периода включительно в формате, совместимом с pandas.
        История с MOEX подгружается с дополнительным месяцем назад для корректных лагов и
        скользящих признаков; расчёт и выдача ограничиваются именно этими датами пользователя.
    end_date : str
        Конец периода включительно в формате, совместимом с pandas.
    initial_capital_rub : float
        Начальный баланс портфеля.
    security : str, default="SBER"
        Тикер инструмента для загрузки из MOEX ISS.
    artifacts_dir : str or Path, default="models"
        Директория с metadata и сериализованными моделями.
    output_dir : str or Path or None, optional
        Необязательная директория для сохранения построенных графиков.
    moex_config : MoexIssConfig or None, optional
        Необязательная конфигурация запросов к MOEX ISS.
    """

    start_date: str
    end_date: str
    initial_capital_rub: float
    security: str = "SBER"
    artifacts_dir: str | Path = "models"
    output_dir: str | Path | None = None
    moex_config: MoexIssConfig | None = None


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Структурированный результат production-пайплайна.

    Parameters
    ----------
    config : PipelineConfig
        Итоговая конфигурация пайплайна.
    market_data : pd.DataFrame
        Рыночные данные только в окне ``start_date``–``end_date`` из конфигурации (без месяца прогрева).
    evaluation : EvaluationResult
        Результат оценки со сводками, кривыми и графиками.
    chart_paths : dict[str, Path]
        Пути к сохранённым графикам по имени стратегии.
    """

    config: PipelineConfig
    market_data: pd.DataFrame
    evaluation: EvaluationResult
    chart_paths: dict[str, Path]

    @property
    def summary_json(self) -> dict[str, dict[str, float]]:
        """Возвращает JSON-готовую сводку по прибыли.

        Returns
        -------
        dict[str, dict[str, float]]
            Сводка по прибыли, сгруппированная по стратегиям.
        """

        return self.evaluation.summary_json

    @property
    def strategy_results(self) -> dict[str, StrategyEvaluationResult]:
        """Возвращает детали оценки по каждой стратегии.

        Returns
        -------
        dict[str, StrategyEvaluationResult]
            Сводки, кривые и графики на уровне стратегии.
        """

        return self.evaluation.strategies

    @property
    def charts(self) -> dict[str, Figure]:
        """Возвращает построенные графики без обязательного сохранения на диск.

        Returns
        -------
        dict[str, Figure]
            Объекты Matplotlib Figure по имени стратегии.
        """

        return {
            strategy_name: strategy_result.chart
            for strategy_name, strategy_result in self.evaluation.strategies.items()
        }


class ProductionPipeline:
    """Координирует загрузку данных, построение фичей и оценку стратегий."""

    def __init__(
        self,
        data_loader: MoexIssDataLoader | None = None,
        feature_extractor: FeatureExtractor | None = None,
        strategy_evaluator: StrategyEvaluator | None = None,
    ) -> None:
        """Инициализирует пайплайн.

        Parameters
        ----------
        data_loader : MoexIssDataLoader or None, optional
            Загрузчик рыночных данных.
        feature_extractor : FeatureExtractor or None, optional
            Сервис построения фичей.
        strategy_evaluator : StrategyEvaluator or None, optional
            Сервис оценки стратегий.
        """

        self._data_loader = data_loader or MoexIssDataLoader()
        self._feature_extractor = feature_extractor or FeatureExtractor()
        self._strategy_evaluator = strategy_evaluator or StrategyEvaluator()

    def run(self, config: PipelineConfig) -> PipelineResult:
        """Запускает production-пайплайн целиком.

        Parameters
        ----------
        config : PipelineConfig
            Конфигурация пайплайна.

        Returns
        -------
        PipelineResult
            Структурированный результат выполнения пайплайна.
        """

        effective_loader = self._data_loader
        if config.moex_config is not None:
            effective_loader = MoexIssDataLoader(config=config.moex_config)

        user_start_iso = _to_iso_date(config.start_date)
        user_end_iso = _to_iso_date(config.end_date)
        fetch_start_iso = _moex_fetch_start_date(
            user_start_iso,
            lookback_months=MOEX_HISTORY_LOOKBACK_MONTHS,
        )

        market_data_full = effective_loader.load_history(
            security=config.security,
            start_date=fetch_start_iso,
            end_date=user_end_iso,
        )
        registry = ArtifactRegistry.from_directory(config.artifacts_dir)
        prepared_full = self._feature_extractor.build_datasets(market_data_full, registry)
        prepared_datasets = _trim_prepared_datasets_to_window(
            prepared_full,
            start_iso=user_start_iso,
            end_iso=user_end_iso,
        )
        evaluation = self._strategy_evaluator.evaluate(
            datasets=prepared_datasets,
            initial_capital_rub=config.initial_capital_rub,
        )
        market_data = _trim_market_data_to_window(
            market_data_full,
            start_iso=user_start_iso,
            end_iso=user_end_iso,
        )
        chart_paths = self._save_charts(evaluation, output_dir=config.output_dir)
        return PipelineResult(
            config=config,
            market_data=market_data,
            evaluation=evaluation,
            chart_paths=chart_paths,
        )

    def _save_charts(
        self,
        evaluation: EvaluationResult,
        *,
        output_dir: str | Path | None,
    ) -> dict[str, Path]:
        """Сохраняет построенные графики, если задана выходная директория.

        Parameters
        ----------
        evaluation : EvaluationResult
            Результат оценки с построенными графиками.
        output_dir : str or Path or None
            Необязательная целевая директория.

        Returns
        -------
        dict[str, Path]
            Пути к сохранённым графикам. Если сохранение отключено, возвращается пустой словарь.
        """

        if output_dir is None:
            return {}

        target_dir = Path(output_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: dict[str, Path] = {}
        for strategy_name, strategy_result in evaluation.strategies.items():
            path = target_dir / f"{strategy_name}_portfolio_curve.png"
            strategy_result.chart.savefig(path, dpi=200, bbox_inches="tight")
            saved_paths[strategy_name] = path
        return saved_paths


def run_pipeline(
    *,
    start_date: str,
    end_date: str,
    initial_capital_rub: float,
    security: str = "SBER",
    artifacts_dir: str | Path = "models",
    output_dir: str | Path | None = None,
    moex_config: MoexIssConfig | None = None,
) -> PipelineResult:
    """Запускает production-пайплайн через модульный API.

    Parameters
    ----------
    start_date : str
        Начало периода включительно в формате, совместимом с pandas.
        К MOEX уходит запрос с датой ``from`` на месяц раньше (см. ``MOEX_HISTORY_LOOKBACK_MONTHS``);
        кривые и сводка строятся только в запрошенном окне.
    end_date : str
        Конец периода включительно в формате, совместимом с pandas.
    initial_capital_rub : float
        Начальный баланс портфеля.
    security : str, default="SBER"
        Тикер инструмента для загрузки из MOEX ISS.
    artifacts_dir : str or Path, default="models"
        Директория с metadata и сериализованными моделями.
    output_dir : str or Path or None, optional
        Необязательная директория для сохранения построенных графиков.
    moex_config : MoexIssConfig or None, optional
        Необязательная конфигурация запросов к MOEX ISS.

    Returns
    -------
    PipelineResult
        Структурированный результат выполнения пайплайна.
    """

    pipeline = ProductionPipeline()
    config = PipelineConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital_rub=initial_capital_rub,
        security=security,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        moex_config=moex_config,
    )
    return pipeline.run(config)


__all__ = [
    "MOEX_HISTORY_LOOKBACK_MONTHS",
    "PipelineConfig",
    "PipelineResult",
    "ProductionPipeline",
    "run_pipeline",
]

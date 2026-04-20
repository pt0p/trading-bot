"""Публичный orchestration API для production-торгового пайплайна."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from bot.data_loader import MoexIssConfig, MoexIssDataLoader
from bot.eval import EvaluationResult, StrategyEvaluationResult, StrategyEvaluator
from bot.feature_extractor import ArtifactRegistry, FeatureExtractor


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Конфигурация production-пайплайна.

    Parameters
    ----------
    start_date : str
        Начало периода включительно в формате, совместимом с pandas.
    end_date : str
        Конец периода включительно в формате, совместимом с pandas.
    initial_capital_rub : float
        Начальный баланс портфеля.
    security : str, default="SBER"
        Тикер инструмента для загрузки из MOEX ISS.
    artifacts_dir : str or Path, default="experiments_2/best_models"
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
        Загруженные и нормализованные рыночные данные.
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

        market_data = effective_loader.load_history(
            security=config.security,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        registry = ArtifactRegistry.from_directory(config.artifacts_dir)
        prepared_datasets = self._feature_extractor.build_datasets(market_data, registry)
        evaluation = self._strategy_evaluator.evaluate(
            datasets=prepared_datasets,
            initial_capital_rub=config.initial_capital_rub,
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
    end_date : str
        Конец периода включительно в формате, совместимом с pandas.
    initial_capital_rub : float
        Начальный баланс портфеля.
    security : str, default="SBER"
        Тикер инструмента для загрузки из MOEX ISS.
    artifacts_dir : str or Path, default="experiments_2/best_models"
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


__all__ = ["PipelineConfig", "PipelineResult", "ProductionPipeline", "run_pipeline"]

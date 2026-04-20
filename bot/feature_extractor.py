"""Построение фичей и реестр артефактов для production-пайплайна."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from bot.data_loader import REQUIRED_MARKET_COLUMNS

ModelName = Literal["linear_regression", "logistic_regression", "catboost_classifier"]

MODEL_ORDER: tuple[ModelName, ...] = (
    "linear_regression",
    "logistic_regression",
    "catboost_classifier",
)


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Параметры генерации фичей, воспроизведённые из обучения.

    Parameters
    ----------
    name : str
        Человекочитаемый идентификатор конфигурации.
    n_lags : int
        Число лаговых признаков по цене закрытия.
    include_ohlc : bool
        Нужно ли включать OHLC-значения текущего шага.
    include_current_return : bool
        Нужно ли включать лаговый признак доходности текущего шага.
    include_return_lags : bool
        Нужно ли включать лаговые признаки доходности.
    include_rolling_stats : bool
        Нужно ли включать признаки на основе скользящей статистики.
    rolling_window : int or None, optional
        Размер окна для rolling-признаков.
    """

    name: str
    n_lags: int
    include_ohlc: bool
    include_current_return: bool
    include_return_lags: bool
    include_rolling_stats: bool
    rolling_window: int | None = None


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Разобранные metadata и пути к модели для обученного артефакта.

    Parameters
    ----------
    model_name : ModelName
        Каноническое имя модели.
    metadata_path : Path
        Путь к JSON-файлу с metadata.
    model_path : Path
        Путь к сериализованному бинарному файлу модели.
    feature_config_name : str
        Имя конфигурации фичей, использованной при обучении.
    feature_columns : tuple[str, ...]
        Упорядоченный список фичей, который ожидает обученная модель.
    decision_threshold : float or None
        Порог принятия решения, сохранённый при обучении классификатора.
    """

    model_name: ModelName
    metadata_path: Path
    model_path: Path
    feature_config_name: str
    feature_columns: tuple[str, ...]
    decision_threshold: float | None


@dataclass(frozen=True, slots=True)
class PreparedModelDataset:
    """Подготовленный для инференса датасет одной модели.

    Parameters
    ----------
    artifact : ModelArtifact
        Разобранные metadata артефакта.
    dataset : pd.DataFrame
        Готовый к бэктесту датасет с колонками ``time``, фичами,
        ``return_t`` и ``stock_price``.
    """

    artifact: ModelArtifact
    dataset: pd.DataFrame

    @property
    def feature_frame(self) -> pd.DataFrame:
        """Возвращает фичи в точном порядке, ожидаемом артефактом.

        Returns
        -------
        pd.DataFrame
            DataFrame с фичами в правильном порядке.
        """

        return self.dataset.loc[:, list(self.artifact.feature_columns)].copy()


TRAINING_FEATURE_CONFIGS: dict[str, FeatureConfig] = {
    "notebook_lag3": FeatureConfig(
        name="notebook_lag3",
        n_lags=3,
        include_ohlc=True,
        include_current_return=True,
        include_return_lags=False,
        include_rolling_stats=False,
    ),
    "price_lag5": FeatureConfig(
        name="price_lag5",
        n_lags=5,
        include_ohlc=True,
        include_current_return=True,
        include_return_lags=False,
        include_rolling_stats=False,
    ),
    "price_and_return_lag7": FeatureConfig(
        name="price_and_return_lag7",
        n_lags=7,
        include_ohlc=True,
        include_current_return=True,
        include_return_lags=True,
        include_rolling_stats=False,
    ),
    "enriched_lag10": FeatureConfig(
        name="enriched_lag10",
        n_lags=10,
        include_ohlc=True,
        include_current_return=True,
        include_return_lags=True,
        include_rolling_stats=True,
        rolling_window=5,
    ),
}


class ArtifactRegistry:
    """Читает и валидирует артефакты моделей из директории."""

    def __init__(self, artifacts: dict[ModelName, ModelArtifact]) -> None:
        """Инициализирует реестр.

        Parameters
        ----------
        artifacts : dict[ModelName, ModelArtifact]
            Словарь с разобранными артефактами.
        """

        self._artifacts = artifacts

    @classmethod
    def from_directory(cls, artifacts_dir: str | Path) -> ArtifactRegistry:
        """Загружает и валидирует metadata для всех обязательных моделей.

        Parameters
        ----------
        artifacts_dir : str or Path
            Директория, содержащая metadata и сериализованные файлы моделей.

        Returns
        -------
        ArtifactRegistry
            Реестр с провалидированными артефактами.
        """

        directory = Path(artifacts_dir).resolve()
        if not directory.exists() or not directory.is_dir():
            msg = f"Artifacts directory does not exist: {directory}"
            raise ValueError(msg)

        artifacts: dict[ModelName, ModelArtifact] = {}
        for model_name in MODEL_ORDER:
            metadata_path = directory / f"{model_name}.metadata.json"
            if not metadata_path.exists():
                msg = f"Metadata file is missing for {model_name!r}: {metadata_path}"
                raise ValueError(msg)
            artifacts[model_name] = cls._load_artifact(metadata_path=metadata_path, artifacts_dir=directory)
        return cls(artifacts)

    @classmethod
    def _load_artifact(cls, *, metadata_path: Path, artifacts_dir: Path) -> ModelArtifact:
        """Читает один файл metadata и вычисляет путь к бинарному артефакту.

        Parameters
        ----------
        metadata_path : Path
            Путь к JSON-файлу metadata.
        artifacts_dir : Path
            Базовая директория артефактов.

        Returns
        -------
        ModelArtifact
            Провалидированное описание артефакта.
        """

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_name = cls._parse_model_name(payload.get("model"))
        feature_config_name = cls._parse_feature_config_name(payload.get("feature_config_name"))
        feature_columns = cls._parse_feature_columns(payload.get("feature_columns"))
        decision_threshold = cls._parse_decision_threshold(payload.get("decision_threshold"))
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            msg = f"Metadata file {metadata_path} does not contain an 'artifacts' block"
            raise ValueError(msg)

        raw_model_path = artifacts.get("model_path")
        if not isinstance(raw_model_path, str) or not raw_model_path.strip():
            msg = f"Metadata file {metadata_path} contains invalid 'artifacts.model_path'"
            raise ValueError(msg)

        resolved_model_path = cls._resolve_artifact_path(
            artifacts_dir=artifacts_dir,
            metadata_path=metadata_path,
            raw_artifact_path=raw_model_path,
        )
        if not resolved_model_path.exists():
            msg = (
                f"Serialized model for {model_name!r} was not found. "
                f"Expected {resolved_model_path}"
            )
            raise ValueError(msg)

        return ModelArtifact(
            model_name=model_name,
            metadata_path=metadata_path,
            model_path=resolved_model_path,
            feature_config_name=feature_config_name,
            feature_columns=feature_columns,
            decision_threshold=decision_threshold,
        )

    @classmethod
    def _resolve_artifact_path(
        cls,
        *,
        artifacts_dir: Path,
        metadata_path: Path,
        raw_artifact_path: str,
    ) -> Path:
        """Разрешает путь к артефакту из metadata переносимым способом.

        Parameters
        ----------
        artifacts_dir : Path
            Директория со всеми файлами артефактов.
        metadata_path : Path
            Файл metadata, который сейчас обрабатывается.
        raw_artifact_path : str
            Строка пути, сохранённая в metadata.

        Returns
        -------
        Path
            Итоговый вычисленный путь.
        """

        candidate = Path(raw_artifact_path)
        if candidate.is_absolute():
            return candidate

        local_candidate = artifacts_dir / candidate.name
        if local_candidate.exists():
            return local_candidate.resolve()

        sibling_candidate = metadata_path.parent / candidate
        if sibling_candidate.exists():
            return sibling_candidate.resolve()

        return local_candidate.resolve()

    @classmethod
    def _parse_model_name(cls, value: object) -> ModelName:
        """Проверяет имя модели, сохранённое в metadata.

        Parameters
        ----------
        value : object
            Сырое поле из JSON.

        Returns
        -------
        ModelName
            Каноническое имя модели.
        """

        if value not in MODEL_ORDER:
            msg = f"Unsupported model name in metadata: {value!r}"
            raise ValueError(msg)
        return value

    @classmethod
    def _parse_feature_config_name(cls, value: object) -> str:
        """Проверяет имя feature-конфигурации, сохранённое в metadata.

        Parameters
        ----------
        value : object
            Сырое поле из JSON.

        Returns
        -------
        str
            Идентификатор feature-конфигурации.
        """

        if not isinstance(value, str) or value not in TRAINING_FEATURE_CONFIGS:
            msg = f"Unsupported feature configuration in metadata: {value!r}"
            raise ValueError(msg)
        return value

    @classmethod
    def _parse_feature_columns(cls, value: object) -> tuple[str, ...]:
        """Проверяет упорядоченный список фичей из metadata.

        Parameters
        ----------
        value : object
            Сырое поле из JSON.

        Returns
        -------
        tuple[str, ...]
            Упорядоченный список фичей.
        """

        if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
            msg = f"Invalid feature_columns payload: {value!r}"
            raise ValueError(msg)
        return tuple(value)

    @classmethod
    def _parse_decision_threshold(cls, value: object) -> float | None:
        """Нормализует порог принятия решения из metadata.

        Parameters
        ----------
        value : object
            Сырое поле из JSON.

        Returns
        -------
        float or None
            Разобранный порог или ``None`` для регрессионных моделей.
        """

        if value is None:
            return None
        threshold = float(value)
        if not 0.0 < threshold < 1.0:
            msg = f"decision_threshold must be strictly between 0 and 1, got {threshold}"
            raise ValueError(msg)
        return threshold

    def get(self, model_name: ModelName) -> ModelArtifact:
        """Возвращает один артефакт по имени модели.

        Parameters
        ----------
        model_name : ModelName
            Запрошенный идентификатор модели.

        Returns
        -------
        ModelArtifact
            Разобранное описание артефакта.
        """

        return self._artifacts[model_name]

    def items(self) -> list[tuple[ModelName, ModelArtifact]]:
        """Возвращает артефакты в каноническом порядке моделей.

        Returns
        -------
        list[tuple[ModelName, ModelArtifact]]
            Упорядоченный список артефактов.
        """

        return [(model_name, self._artifacts[model_name]) for model_name in MODEL_ORDER]


class FeatureExtractor:
    """Строит готовые к инференсу датасеты фичей из сырых рыночных данных."""

    def build_datasets(
        self,
        market_data: pd.DataFrame,
        registry: ArtifactRegistry,
    ) -> dict[ModelName, PreparedModelDataset]:
        """Строит датасеты для всех моделей из одного набора рыночных данных.

        Parameters
        ----------
        market_data : pd.DataFrame
            Сырые дневные рыночные данные из MOEX ISS.
        registry : ArtifactRegistry
            Провалидированный реестр артефактов.

        Returns
        -------
        dict[ModelName, PreparedModelDataset]
            Готовые к бэктесту датасеты, сгруппированные по имени модели.
        """

        normalized = self._normalize_market_data(market_data)
        prepared: dict[ModelName, PreparedModelDataset] = {}
        for model_name, artifact in registry.items():
            feature_config = TRAINING_FEATURE_CONFIGS[artifact.feature_config_name]
            full_feature_frame = self._build_feature_frame(normalized, feature_config)
            dataset = self._build_backtest_dataset(
                market_data=normalized,
                feature_frame=full_feature_frame,
                feature_columns=artifact.feature_columns,
            )
            prepared[model_name] = PreparedModelDataset(artifact=artifact, dataset=dataset)
        return prepared

    def _normalize_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Проверяет и нормализует сырой DataFrame с рыночными данными.

        Parameters
        ----------
        market_data : pd.DataFrame
            Сырые входные рыночные данные.

        Returns
        -------
        pd.DataFrame
            Отсортированные и приведённые к нужным типам рыночные данные.
        """

        if market_data.empty:
            msg = "market_data must not be empty"
            raise ValueError(msg)

        missing_columns = [column for column in REQUIRED_MARKET_COLUMNS if column not in market_data.columns]
        if missing_columns:
            msg = f"market_data is missing required columns: {missing_columns}"
            raise ValueError(msg)

        normalized = market_data.loc[:, list(REQUIRED_MARKET_COLUMNS)].copy()
        normalized["TRADEDATE"] = pd.to_datetime(normalized["TRADEDATE"], errors="coerce").dt.strftime("%Y-%m-%d")
        if normalized["TRADEDATE"].isna().any():
            msg = "market_data contains invalid TRADEDATE values"
            raise ValueError(msg)
        if normalized["TRADEDATE"].duplicated().any():
            msg = "market_data contains duplicate TRADEDATE values"
            raise ValueError(msg)

        for column in REQUIRED_MARKET_COLUMNS[1:]:
            numeric_values = pd.to_numeric(normalized[column], errors="coerce")
            if numeric_values.isna().any():
                msg = f"market_data contains invalid numeric values in {column!r}"
                raise ValueError(msg)
            normalized[column] = numeric_values.astype(float)

        return normalized.sort_values("TRADEDATE").reset_index(drop=True)

    def _build_feature_frame(self, market_data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """Воспроизводит обучающие фичи для одной конфигурации.

        Parameters
        ----------
        market_data : pd.DataFrame
            Нормализованные рыночные данные.
        config : FeatureConfig
            Выбранная обучающая конфигурация.

        Returns
        -------
        pd.DataFrame
            DataFrame со всеми построенными фичами до фильтрации колонок.
        """

        close = market_data["LEGALCLOSEPRICE"].astype(float)
        log_return = np.log(close / close.shift(1))
        features = pd.DataFrame(index=market_data.index)

        if config.include_ohlc:
            features["open"] = market_data["OPEN"].to_numpy(dtype=float, copy=False)
            features["low"] = market_data["LOW"].to_numpy(dtype=float, copy=False)
            features["high"] = market_data["HIGH"].to_numpy(dtype=float, copy=False)

        for lag in range(1, config.n_lags + 1):
            features[f"close_lag_{lag}"] = close.shift(lag).to_numpy(dtype=float, copy=False)

        if config.include_current_return:
            features["log_return_lag_1"] = log_return.shift(1).to_numpy(dtype=float, copy=False)

        if config.include_return_lags:
            for lag in range(1, config.n_lags + 1):
                features[f"return_lag_{lag}"] = log_return.shift(lag).to_numpy(dtype=float, copy=False)

        if config.include_rolling_stats and config.rolling_window is not None:
            rolling_mean = close.rolling(config.rolling_window).mean()
            rolling_std = log_return.rolling(config.rolling_window).std()
            features[f"close_vs_mean_{config.rolling_window}"] = (
                (close - rolling_mean) / rolling_mean
            ).to_numpy(dtype=float, copy=False)
            features[f"return_std_{config.rolling_window}"] = rolling_std.to_numpy(dtype=float, copy=False)

        return features

    def _build_backtest_dataset(
        self,
        *,
        market_data: pd.DataFrame,
        feature_frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
    ) -> pd.DataFrame:
        """Строит готовый к бэктесту датасет для одного артефакта.

        Parameters
        ----------
        market_data : pd.DataFrame
            Нормализованные рыночные данные.
        feature_frame : pd.DataFrame
            Все сгенерированные фичи.
        feature_columns : tuple[str, ...]
            Упорядоченный список фичей, требуемых моделью.

        Returns
        -------
        pd.DataFrame
            Очищенный датасет с фичами, ``time``, ``return_t`` и
            ``stock_price``.
        """

        missing_columns = [column for column in feature_columns if column not in feature_frame.columns]
        if missing_columns:
            msg = f"Generated features do not match metadata. Missing columns: {missing_columns}"
            raise ValueError(msg)

        selected_features = feature_frame.loc[:, list(feature_columns)].copy()
        assembled = selected_features.copy()
        assembled.insert(0, "time", market_data["TRADEDATE"].to_numpy(copy=False))
        assembled["return_t"] = self._continuous_target(market_data).to_numpy(dtype=float, copy=False)
        assembled["stock_price"] = self._price_target(market_data).to_numpy(dtype=float, copy=False)
        assembled = assembled.dropna().reset_index(drop=True)
        if assembled.empty:
            msg = "Feature extraction produced an empty backtest dataset for at least one model"
            raise ValueError(msg)
        return assembled

    def _continuous_target(self, market_data: pd.DataFrame) -> pd.Series:
        """Строит ряд лог-доходности следующего шага.

        Parameters
        ----------
        market_data : pd.DataFrame
            Нормализованные рыночные данные.

        Returns
        -------
        pd.Series
            Лог-доходность следующего шага.

        Notes
        -----
        Для последней календарной строки ряда значение отсутствует, потому что
        доходность считается через следующий бар. Такая строка затем отфильтровывается
        на этапе сборки backtest-датасета.
        """

        close = market_data["LEGALCLOSEPRICE"].astype(float)
        return pd.Series(np.log(close.shift(-1) / close), index=market_data.index, name="return_t")

    def _price_target(self, market_data: pd.DataFrame) -> pd.Series:
        """Строит ряд цены закрытия следующего шага.

        Parameters
        ----------
        market_data : pd.DataFrame
            Нормализованные рыночные данные.

        Returns
        -------
        pd.Series
            Цена закрытия следующего шага.
        """

        close = market_data["LEGALCLOSEPRICE"].astype(float)
        return pd.Series(close.shift(-1), index=market_data.index, name="stock_price")


__all__ = [
    "ArtifactRegistry",
    "FeatureConfig",
    "FeatureExtractor",
    "MODEL_ORDER",
    "ModelArtifact",
    "ModelName",
    "PreparedModelDataset",
    "TRAINING_FEATURE_CONFIGS",
]

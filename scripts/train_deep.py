from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "reports"
    / "eda"
    / "daily_eda_dataset.csv"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "deep_learning"
)

ARTIFACT_DIR = (
    ROOT
    / "artifacts"
    / "deep_learning"
)


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(2)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 20

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class AQIMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(
                128,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(
                64,
                32,
            ),
            nn.ReLU(),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"EDA dataset not found: {DATA_PATH}\n"
            "Run: python scripts/eda.py"
        )

    frame = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    required = {
        "city",
        "local_date",
        "aqi",
    }

    missing = required.difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame["local_date"] = pd.to_datetime(
        frame["local_date"],
        errors="coerce",
    )

    frame["aqi"] = pd.to_numeric(
        frame["aqi"],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "city",
            "local_date",
            "aqi",
        ]
    )

    frame = (
        frame
        .sort_values(
            [
                "city",
                "local_date",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------------
    # Target:
    # Today's features -> tomorrow's AQI.
    # --------------------------------------------------------------

    frame["target_aqi_next_day"] = (
        frame
        .groupby("city")["aqi"]
        .shift(-1)
    )

    frame["target_date"] = (
        frame
        .groupby("city")["local_date"]
        .shift(-1)
    )

    frame = frame.dropna(
        subset=[
            "target_aqi_next_day",
            "target_date",
        ]
    )

    return frame.reset_index(
        drop=True
    )


def chronological_split(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    dates = np.array(
        sorted(
            frame[
                "local_date"
            ]
            .dropna()
            .unique()
        )
    )

    if len(dates) < 30:
        raise ValueError(
            "Not enough unique dates "
            "for chronological train/validation/test split."
        )

    train_end_index = int(
        len(dates)
        * TRAIN_FRACTION
    )

    validation_end_index = int(
        len(dates)
        * (
            TRAIN_FRACTION
            + VALIDATION_FRACTION
        )
    )

    train_end_date = dates[
        train_end_index - 1
    ]

    validation_end_date = dates[
        validation_end_index - 1
    ]

    train = frame[
        frame["local_date"]
        <= train_end_date
    ].copy()

    validation = frame[
        (
            frame["local_date"]
            > train_end_date
        )
        & (
            frame["local_date"]
            <= validation_end_date
        )
    ].copy()

    test = frame[
        frame["local_date"]
        > validation_end_date
    ].copy()

    if (
        train.empty
        or validation.empty
        or test.empty
    ):
        raise RuntimeError(
            "Chronological split produced "
            "an empty partition."
        )

    print(
        "\nChronological split:",
        flush=True,
    )

    print(
        f"Train: {len(train):,} rows | "
        f"{train['local_date'].min().date()} "
        f"to {train['local_date'].max().date()}",
        flush=True,
    )

    print(
        f"Validation: {len(validation):,} rows | "
        f"{validation['local_date'].min().date()} "
        f"to {validation['local_date'].max().date()}",
        flush=True,
    )

    print(
        f"Test: {len(test):,} rows | "
        f"{test['local_date'].min().date()} "
        f"to {test['local_date'].max().date()}",
        flush=True,
    )

    return (
        train,
        validation,
        test,
    )


# ------------------------------------------------------------------
# Feature preparation
# ------------------------------------------------------------------

def prepare_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    dict[str, float],
    dict[str, float],
]:
    combined = pd.concat(
        [
            train.assign(
                __split="train"
            ),
            validation.assign(
                __split="validation"
            ),
            test.assign(
                __split="test"
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    # One-hot encode city.
    city_dummies = pd.get_dummies(
        combined["city"],
        prefix="city",
        dtype=float,
    )

    combined = pd.concat(
        [
            combined,
            city_dummies,
        ],
        axis=1,
    )

    excluded = {
        "city",
        "local_date",
        "target_date",
        "target_aqi_next_day",
        "season",
        "__split",
    }

    candidate_columns: list[str] = []

    for column in combined.columns:
        if column in excluded:
            continue

        converted = pd.to_numeric(
            combined[column],
            errors="coerce",
        )

        # Keep columns having at least some
        # numeric information.
        if converted.notna().any():
            combined[column] = converted

            candidate_columns.append(
                column
            )

    train_mask = (
        combined["__split"]
        == "train"
    )

    medians: dict[str, float] = {}

    usable_columns: list[str] = []

    for column in candidate_columns:
        train_values = pd.to_numeric(
            combined.loc[
                train_mask,
                column,
            ],
            errors="coerce",
        )

        median = train_values.median()

        if pd.isna(median):
            continue

        medians[column] = float(
            median
        )

        combined[column] = (
            pd.to_numeric(
                combined[column],
                errors="coerce",
            )
            .fillna(median)
        )

        # Remove constant features.
        if (
            combined.loc[
                train_mask,
                column,
            ].std()
            <= 1e-12
        ):
            continue

        usable_columns.append(
            column
        )

    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for column in usable_columns:
        values = combined.loc[
            train_mask,
            column,
        ]

        mean = float(
            values.mean()
        )

        std = float(
            values.std()
        )

        if (
            not np.isfinite(std)
            or std <= 1e-12
        ):
            std = 1.0

        means[column] = mean
        stds[column] = std

        combined[column] = (
            combined[column]
            - mean
        ) / std

    def extract(
        split_name: str,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        subset = combined[
            combined["__split"]
            == split_name
        ]

        x = (
            subset[
                usable_columns
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        y = (
            subset[
                "target_aqi_next_day"
            ]
            .to_numpy(
                dtype=np.float32
            )
        )

        return x, y

    x_train, y_train = extract(
        "train"
    )

    x_validation, y_validation = extract(
        "validation"
    )

    x_test, y_test = extract(
        "test"
    )

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        usable_columns,
        means,
        stds,
    )


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[
    AQIMLP,
    list[float],
    list[float],
]:
    device = torch.device("cpu")

    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )

    validation_x = torch.from_numpy(
        x_validation
    ).to(device)

    validation_y = torch.from_numpy(
        y_validation
    ).to(device)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )

    model = AQIMLP(
        input_size=x_train.shape[1],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    train_losses: list[float] = []
    validation_losses: list[float] = []

    best_validation_loss = float(
        "inf"
    )

    best_state = None

    epochs_without_improvement = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        model.train()

        batch_losses: list[float] = []

        for (
            batch_x,
            batch_y,
        ) in train_loader:
            batch_x = batch_x.to(
                device
            )

            batch_y = batch_y.to(
                device
            )

            optimizer.zero_grad()

            prediction = model(
                batch_x
            )

            loss = criterion(
                prediction,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            batch_losses.append(
                float(loss.item())
            )

        train_loss = float(
            np.mean(batch_losses)
        )

        model.eval()

        with torch.no_grad():
            validation_prediction = model(
                validation_x
            )

            validation_loss = float(
                criterion(
                    validation_prediction,
                    validation_y,
                ).item()
            )

        train_losses.append(
            train_loss
        )

        validation_losses.append(
            validation_loss
        )

        if (
            validation_loss
            < best_validation_loss
            - 1e-4
        ):
            best_validation_loss = (
                validation_loss
            )

            best_state = {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epoch == 1
            or epoch % 10 == 0
        ):
            print(
                f"Epoch {epoch:03d} | "
                f"train MSE={train_loss:.3f} | "
                f"val MSE={validation_loss:.3f}",
                flush=True,
            )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            print(
                f"Early stopping at epoch {epoch}",
                flush=True,
            )

            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce "
            "a valid model state."
        )

    model.load_state_dict(
        best_state
    )

    return (
        model,
        train_losses,
        validation_losses,
    )


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def predict(
    model: AQIMLP,
    x: np.ndarray,
) -> np.ndarray:
    model.eval()

    with torch.no_grad():
        tensor = torch.from_numpy(
            x
        )

        prediction = model(
            tensor
        ).cpu().numpy()

    return prediction.astype(
        np.float64
    )


def metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    return {
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "mae": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "r2": float(
            r2_score(
                actual,
                predicted,
            )
        ),
        "bias": float(
            np.mean(
                predicted
                - actual
            )
        ),
    }


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def save_loss_plot(
    train_losses: list[float],
    validation_losses: list[float],
) -> None:
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(
        train_losses,
        label="Training",
    )

    ax.plot(
        validation_losses,
        label="Validation",
    )

    ax.set_title(
        "Deep AQI Model Training History"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR
        / "01_training_history.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_prediction_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> None:
    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    ax.scatter(
        actual,
        predicted,
        alpha=0.25,
        s=15,
    )

    lower = float(
        min(
            actual.min(),
            predicted.min(),
        )
    )

    upper = float(
        max(
            actual.max(),
            predicted.max(),
        )
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
    )

    ax.set_title(
        "Deep Model: Predicted vs Actual AQI"
    )

    ax.set_xlabel(
        "Actual Next-Day AQI"
    )

    ax.set_ylabel(
        "Predicted Next-Day AQI"
    )

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR
        / "02_actual_vs_predicted.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Starting AQI deep-learning experiment",
        flush=True,
    )

    frame = load_dataset()

    print(
        f"Usable next-day rows: {len(frame):,}",
        flush=True,
    )

    (
        train,
        validation,
        test,
    ) = chronological_split(
        frame
    )

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        feature_columns,
        means,
        stds,
    ) = prepare_features(
        train,
        validation,
        test,
    )

    print(
        f"\nDeep model input features: "
        f"{len(feature_columns)}",
        flush=True,
    )

    print(
        f"Train matrix: {x_train.shape}",
        flush=True,
    )

    print(
        f"Validation matrix: "
        f"{x_validation.shape}",
        flush=True,
    )

    print(
        f"Test matrix: {x_test.shape}",
        flush=True,
    )

    (
        model,
        train_losses,
        validation_losses,
    ) = train_model(
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    predictions = predict(
        model,
        x_test,
    )

    deep_metrics = metrics(
        y_test,
        predictions,
    )

    # --------------------------------------------------------------
    # Persistence baseline:
    # tomorrow AQI ~= today's AQI
    # --------------------------------------------------------------

    persistence_predictions = (
        pd.to_numeric(
            test["aqi"],
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    persistence_metrics = metrics(
        y_test,
        persistence_predictions,
    )

    print(
        "\nDeep MLP test metrics:",
        flush=True,
    )

    print(
        json.dumps(
            deep_metrics,
            indent=2,
        ),
        flush=True,
    )

    print(
        "\nPersistence baseline metrics:",
        flush=True,
    )

    print(
        json.dumps(
            persistence_metrics,
            indent=2,
        ),
        flush=True,
    )

    # --------------------------------------------------------------
    # Save model
    # --------------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "input_size":
                len(feature_columns),
            "feature_columns":
                feature_columns,
        },
        ARTIFACT_DIR
        / "aqi_next_day_mlp.pt",
    )

    preprocessing = {
        "features":
            feature_columns,
        "means":
            means,
        "stds":
            stds,
    }

    (
        ARTIFACT_DIR
        / "preprocessing.json"
    ).write_text(
        json.dumps(
            preprocessing,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------------

    prediction_frame = pd.DataFrame(
        {
            "city":
                test["city"].values,
            "feature_date":
                test[
                    "local_date"
                ].astype(str).values,
            "target_date":
                test[
                    "target_date"
                ].astype(str).values,
            "actual_aqi":
                y_test,
            "deep_mlp_prediction":
                predictions,
            "persistence_prediction":
                persistence_predictions,
        }
    )

    prediction_frame.to_csv(
        REPORT_DIR
        / "test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "model":
                    "PyTorch MLP",
                **deep_metrics,
            },
            {
                "model":
                    "Persistence baseline",
                **persistence_metrics,
            },
        ]
    )

    comparison.to_csv(
        REPORT_DIR
        / "deep_model_comparison.csv",
        index=False,
    )

    result = {
        "experiment":
            "next_day_aqi_mlp",
        "framework":
            "PyTorch",
        "pytorch_version":
            torch.__version__,
        "device":
            "cpu",
        "target":
            "next-day AQI",
        "split_method":
            "chronological 70/15/15",
        "train_rows":
            int(len(train)),
        "validation_rows":
            int(len(validation)),
        "test_rows":
            int(len(test)),
        "feature_count":
            int(
                len(feature_columns)
            ),
        "deep_model_metrics":
            deep_metrics,
        "persistence_baseline_metrics":
            persistence_metrics,
    }

    (
        REPORT_DIR
        / "deep_learning_metrics.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_loss_plot(
        train_losses,
        validation_losses,
    )

    save_prediction_plot(
        y_test,
        predictions,
    )

    print(
        "\nDeep-learning experiment completed successfully.",
        flush=True,
    )

    print(
        f"Reports: {REPORT_DIR}",
        flush=True,
    )

    print(
        f"Model artifact: {ARTIFACT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
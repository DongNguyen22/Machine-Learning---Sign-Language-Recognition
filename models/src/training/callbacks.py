"""Keras callback factories."""

from __future__ import annotations

from pathlib import Path

from tensorflow import keras


def create_training_callbacks(
    checkpoint_path: Path | str,
    patience: int = 12,
    lr_patience: int = 5,
) -> list[keras.callbacks.Callback]:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-6,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

"""EEG + ECG + EDA + Gaze ten-fold cross-validation."""

from pathlib import Path
from datetime import datetime
import json
import random

import numpy as np
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from cl_drive_dataset_build import (
    ECG_CHANNELS,
    ECG_WINDOW_SAMPLES,
    EDA_CHANNELS,
    EDA_WINDOW_SAMPLES,
    EEG_CHANNELS,
    EEG_WINDOW_SAMPLES,
    GAZE_CHANNELS,
    GAZE_WINDOW_SAMPLES,
    FourModalDataset,
    bundle_paths_exist,
    load_four_modal_processed_bundle,
    OUTPUT_DIR as FOUR_MODAL_DATASET_DIR,
)

from rcfm import RCFM


NUM_CLASSES = 2
BATCH_SIZE = 256
GRADIENT_CLIP_VALUE = 0.5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-4
FUSION_NUM_STAGES = 4

RANDOM_STATE = 3409
NUM_EPOCHS_CV = 150
EARLY_STOPPING_PATIENCE = 30
EARLY_STOPPING_METRIC = "valid_f1"

DEBUG = False
DEBUG_N_EPOCHS = 2

OUTPUT_DIR = Path("./results")
SAVE_MODELS = True


def set_seed(seed: int = 3409):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    gradient_clip_value: float | None = GRADIENT_CLIP_VALUE,
):
    model.train()
    train_loss = 0.0
    correct = 0.0
    n_samples = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        (x_eeg, x_ecg, x_eda, x_gaze), y = batch[0], batch[1]
        x_eeg = x_eeg.to(device)
        x_ecg = x_ecg.to(device)
        x_eda = x_eda.to(device)
        x_gaze = x_gaze.to(device)
        y = y.to(device).long().squeeze()
        if y.dim() == 0:
            y = y.unsqueeze(0)

        optimizer.zero_grad()
        logits = model(x_eeg, x_ecg, x_eda, x_gaze)
        loss = loss_fn(logits, y)
        loss.backward()
        if gradient_clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
        optimizer.step()

        train_loss += loss.item()
        preds = logits.argmax(1).cpu().numpy()
        y_np = y.cpu().numpy()
        correct += (preds == y_np).sum()
        n_samples += y.size(0)
        all_preds.append(preds)
        all_labels.append(y_np)

    scheduler.step()
    train_loss /= max(len(dataloader), 1)
    train_acc = correct / max(n_samples, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    train_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return train_loss, train_acc, train_f1


@torch.no_grad()
def evaluate_model(
    dataloader: DataLoader,
    model: Module,
    loss_fn: Module,
    device: torch.device,
):
    model.eval()
    valid_loss = 0.0
    correct = 0.0
    n_samples = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        (x_eeg, x_ecg, x_eda, x_gaze), y = batch[0], batch[1]
        x_eeg = x_eeg.to(device)
        x_ecg = x_ecg.to(device)
        x_eda = x_eda.to(device)
        x_gaze = x_gaze.to(device)
        y = y.to(device).long().squeeze()
        if y.dim() == 0:
            y = y.unsqueeze(0)

        logits = model(x_eeg, x_ecg, x_eda, x_gaze)
        loss = loss_fn(logits, y)
        valid_loss += loss.item()
        preds = logits.argmax(1).cpu().numpy()
        y_np = y.cpu().numpy()
        correct += (preds == y_np).sum()
        n_samples += y.size(0)
        all_preds.append(preds)
        all_labels.append(y_np)

    valid_loss /= max(len(dataloader), 1)
    valid_acc = correct / max(n_samples, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    valid_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    return valid_loss, valid_acc, valid_f1


def _json_default(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_training_results(results: dict, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=_json_default)


def save_kfold_training_log(fold_results: list, output_path: Path) -> None:
    """Persist per-epoch metrics for every CV fold (same fields as in-memory fold_results)."""
    payload = {
        "folds": [
            {
                "fold": int(r["fold"]),
                "best_epoch": int(r["best_epoch"]),
                "valid_accuracy": float(r["valid_accuracy"]),
                "valid_f1": float(r["valid_f1"]),
                "history": r["history"],
            }
            for r in fold_results
        ],
        "best_fold_by_valid_acc": int(
            max(fold_results, key=lambda r: r["valid_accuracy"])["fold"]
        ),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)


def get_multimodal_labels(dataset: FourModalDataset) -> np.ndarray:
    y = np.zeros(len(dataset), dtype=np.int64)
    for i in range(len(dataset)):
        label = dataset[i][1]
        if hasattr(label, "item"):
            y[i] = int(label.item())
        else:
            y[i] = int(np.asarray(label).flat[0])
    return y


def load_four_modal_dataset():
    """Load NPZ + manifest written by cl_drive_dataset_build.main."""
    data_dir = Path(FOUR_MODAL_DATASET_DIR)
    if not bundle_paths_exist(data_dir):
        raise FileNotFoundError(
            "Preprocessed four-modal bundle not found. Run cl_drive_dataset_build.py first "
            f"(four_modal_manifest.json, four_modal_processed.npz). Expected dir: {data_dir}"
        )
    (
        multimodal_dataset,
        eeg_dataset,
        ecg_dataset,
        eda_dataset,
        gaze_dataset,
        _aligned_keys,
    ) = load_four_modal_processed_bundle(data_dir)
    return multimodal_dataset, eeg_dataset, ecg_dataset, eda_dataset, gaze_dataset


def run_kfold_cv_pytorch(
    multimodal_dataset: FourModalDataset,
    device: torch.device,
    n_folds: int = 10,
    n_epochs: int = NUM_EPOCHS_CV,
    n_eeg_chans: int = len(EEG_CHANNELS),
    n_ecg_chans: int = len(ECG_CHANNELS),
    n_eda_chans: int = len(EDA_CHANNELS),
    n_gaze_chans: int = len(GAZE_CHANNELS),
):
    lr = LEARNING_RATE
    wd = WEIGHT_DECAY
    fns = FUSION_NUM_STAGES

    y_all = get_multimodal_labels(multimodal_dataset)
    indices = np.arange(len(multimodal_dataset))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    best_fold_val_acc = -1.0
    best_fold_history = []
    best_model_state = None

    for fold, (train_idx, valid_idx) in enumerate(skf.split(indices, y_all)):
        set_seed(RANDOM_STATE + fold)
        train_set = Subset(multimodal_dataset, train_idx.tolist())
        valid_set = Subset(multimodal_dataset, valid_idx.tolist())

        train_loader = DataLoader(
            train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False
        )
        valid_loader = DataLoader(
            valid_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        model = RCFM(
            n_eeg_chans=n_eeg_chans,
            n_ecg_chans=n_ecg_chans,
            n_eda_chans=n_eda_chans,
            n_gaze_chans=n_gaze_chans,
            n_eeg_times=EEG_WINDOW_SAMPLES,
            n_ecg_times=ECG_WINDOW_SAMPLES,
            n_eda_times=EDA_WINDOW_SAMPLES,
            n_gaze_times=GAZE_WINDOW_SAMPLES,
            n_outputs=NUM_CLASSES,
            fusion_num_stages=fns,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, n_epochs - 1)
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        history = []
        if EARLY_STOPPING_METRIC == "valid_f1":
            best_metric_this_fold = -1.0
            is_better = lambda cur, best: cur > best
        else:
            best_metric_this_fold = float("inf")
            is_better = lambda cur, best: cur < best
        epochs_without_improvement = 0
        best_epoch_this_fold = 1
        best_model_state_this_fold = None

        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc, train_f1 = train_one_epoch(
                train_loader, model, loss_fn, optimizer, scheduler, device
            )
            valid_loss, valid_acc, valid_f1 = evaluate_model(
                valid_loader, model, loss_fn, device
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_acc": float(train_acc),
                    "test_acc": float(valid_acc),
                    "train_f1": float(train_f1),
                    "test_f1": float(valid_f1),
                    "train_loss": float(train_loss),
                    "valid_loss": float(valid_loss),
                }
            )
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"Fold {fold + 1}/{n_folds} epoch {epoch:03d}/{n_epochs} | "
                    f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} | "
                    f"train_acc={train_acc:.4f} valid_acc={valid_acc:.4f} | "
                    f"train_f1={train_f1:.4f} valid_f1={valid_f1:.4f}"
                )
            cur_metric = valid_f1 if EARLY_STOPPING_METRIC == "valid_f1" else valid_loss
            if is_better(cur_metric, best_metric_this_fold):
                best_metric_this_fold = cur_metric
                best_epoch_this_fold = epoch
                epochs_without_improvement = 0
                best_model_state_this_fold = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

            if (
                EARLY_STOPPING_PATIENCE is not None
                and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
            ):
                print(
                    f"Early stop fold {fold + 1}: epoch={epoch} best_epoch={best_epoch_this_fold} "
                    f"metric={EARLY_STOPPING_METRIC}"
                )
                break

        if best_model_state_this_fold is not None:
            model.load_state_dict(
                {k: v.to(device) for k, v in best_model_state_this_fold.items()}
            )
        best_valid_acc = history[best_epoch_this_fold - 1]["test_acc"]
        best_valid_f1 = history[best_epoch_this_fold - 1]["test_f1"]

        fold_results.append(
            {
                "fold": fold + 1,
                "valid_accuracy": float(best_valid_acc),
                "valid_f1": float(best_valid_f1),
                "best_epoch": best_epoch_this_fold,
                "history": history,
            }
        )
        print(
            f"Fold {fold + 1}/{n_folds} done: best_epoch={best_epoch_this_fold} "
            f"val_acc={best_valid_acc:.4f} val_f1={best_valid_f1:.4f}"
        )
        if best_valid_acc > best_fold_val_acc:
            best_fold_val_acc = best_valid_acc
            best_fold_history = history
            best_model_state = best_model_state_this_fold

    fold_accuracies = [result["valid_accuracy"] for result in fold_results]
    fold_f1s = [result["valid_f1"] for result in fold_results]
    print(
        f"CV mean: acc={float(np.mean(fold_accuracies)):.4f} "
        f"f1_macro={float(np.mean(fold_f1s)):.4f}"
    )
    return {
        "fold_results": fold_results,
        "fold_valid_accuracies": [float(a) for a in fold_accuracies],
        "fold_valid_f1s": [float(f) for f in fold_f1s],
        "best_cv_accuracy": float(np.mean(fold_accuracies)),
        "best_cv_f1": float(np.mean(fold_f1s)),
        "best_fold_history": best_fold_history,
        "best_model_state": best_model_state,
    }


def main():
    set_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dev_s = str(device)
    if device.type == "cuda":
        dev_s = f"{device} ({torch.cuda.get_device_name(0)})"

    multimodal_dataset, eeg_dataset, ecg_dataset, eda_dataset, gaze_dataset = (
        load_four_modal_dataset()
    )
    n_eeg_chans = eeg_dataset[0][0].shape[0]
    n_ecg_chans = ecg_dataset[0][0].shape[0]
    n_eda_chans = len(EDA_CHANNELS)
    n_gaze_chans = gaze_dataset[0][0].shape[0]

    n_folds_run = 10
    n_epochs_run = DEBUG_N_EPOCHS if DEBUG else NUM_EPOCHS_CV
    print(
        f"device={dev_s} | n_samples={len(multimodal_dataset)} "
        f"chans EEG/ECG/EDA/Gaze={n_eeg_chans}/{n_ecg_chans}/{n_eda_chans}/{n_gaze_chans} "
        f"| folds={n_folds_run} epochs={n_epochs_run} lr={LEARNING_RATE}"
    )

    cv_out = run_kfold_cv_pytorch(
        multimodal_dataset,
        device,
        n_folds=n_folds_run,
        n_epochs=n_epochs_run,
        n_eeg_chans=n_eeg_chans,
        n_ecg_chans=n_ecg_chans,
        n_eda_chans=n_eda_chans,
        n_gaze_chans=n_gaze_chans,
    )
    best_cv_score = cv_out["best_cv_accuracy"]
    best_cv_f1 = cv_out.get("best_cv_f1")
    best_model_state = cv_out["best_model_state"]
    fold_valid_accuracies = cv_out["fold_valid_accuracies"]
    fold_valid_f1s = cv_out["fold_valid_f1s"]
    fold_results = cv_out["fold_results"]

    if SAVE_MODELS and best_model_state is not None:
        model_save_path = (
            OUTPUT_DIR / "multimodal_eeg_ecg_eda_gaze_cross_attn_gl_stfb_kfold.pth"
        )
        torch.save(best_model_state, model_save_path)

    log_name = "multimodal_four_modality_training_log_kfold_gl_stfb.json"
    save_kfold_training_log(fold_results, OUTPUT_DIR / log_name)

    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "multimodal_eeg_ecg_eda_gaze_alternating_cross_attn_10_fold_cv_gl_stfb",
        "n_folds": 10,
        "dataset_size": len(multimodal_dataset),
        "training_hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "fusion_num_stages": FUSION_NUM_STAGES,
        },
        "cv_results": {
            "best_cv_accuracy": float(best_cv_score),
            "best_cv_accuracy_percent": float(best_cv_score * 100),
            "fold_valid_accuracies": [float(a) for a in fold_valid_accuracies],
            "fold_valid_f1s": [float(f) for f in fold_valid_f1s],
        },
    }
    if best_cv_f1 is not None:
        results["cv_results"]["best_cv_f1"] = float(best_cv_f1)
        results["cv_results"]["best_cv_f1_percent"] = float(best_cv_f1 * 100)

    results_path = OUTPUT_DIR / "multimodal_four_modality_training_results_kfold_gl_stfb.json"
    save_training_results(results, results_path)
    end = f"Done. CV acc={best_cv_score * 100:.2f}%"
    if best_cv_f1 is not None:
        end += f" f1_macro={best_cv_f1 * 100:.2f}%"
    print(end)


if __name__ == "__main__":
    main()

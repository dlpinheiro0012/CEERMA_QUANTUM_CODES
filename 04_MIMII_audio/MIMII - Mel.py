#%%
import os
import glob
import numpy as np
import librosa
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone


# =====================================================
# CONFIGURAÇÕES
# =====================================================

DATASET_PATH = "C:\\Users\\Arthur\\Downloads\\6_dB_pump\\pump"

SAMPLE_RATE = 16000
WINDOW_SECONDS = 2
HOP_SECONDS = 2

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

N_SPLITS = 5
RANDOM_STATE = 42
FIT_PATIENCE = 5     # paciência do EarlyStopping dentro do model.fit() da CNN
EPOCHS = 60
BATCH_SIZE = 16

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =====================================================
# MELHORES PARÂMETROS JÁ ENCONTRADOS (sem grid search)
# =====================================================

SVM_PARAMS = {"C": 10, "gamma": "scale"}
CNN_PARAMS = {"filters": 16, "kernel_size": 5, "dense_units": 128}


# =====================================================
# MEL-SPECTROGRAM FEATURES
# =====================================================

def extract_melspec_features(signal, sr):
    """Mel-spectrogram (dB) do segmento: matriz N_MELS x frames."""
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db  # shape: (N_MELS, n_frames)


def segment_signal(signal, sr, window_sec, hop_sec):
    window = int(window_sec * sr)
    hop = int(hop_sec * sr)
    segments = []

    for start in range(0, len(signal) - window + 1, hop):
        segments.append(signal[start:start + window])

    return segments


# =====================================================
# LOADER
# =====================================================

def load_file_list(dataset_path):
    file_paths = []
    labels = []

    ids = os.listdir(dataset_path)

    for machine_id in ids:
        id_path = os.path.join(dataset_path, machine_id)

        normal_files = glob.glob(os.path.join(id_path, "normal", "*.wav"))
        abnormal_files = glob.glob(os.path.join(id_path, "abnormal", "*.wav"))

        for f in normal_files:
            file_paths.append(f)
            labels.append(0)

        for f in abnormal_files:
            file_paths.append(f)
            labels.append(1)

    return np.array(file_paths), np.array(labels)


def extract_melspec_from_files(file_list):
    """Extrai o Mel-spectrogram de cada segmento dos arquivos."""
    X = []
    y = []

    for file_path, label in tqdm(file_list, leave=False):

        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        segments = segment_signal(signal, sr, WINDOW_SECONDS, HOP_SECONDS)

        for seg in segments:
            X.append(extract_melspec_features(seg, sr))
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y


def prepare_folds(file_paths, labels):
    """Split por arquivo (StratifiedKFold) antes da segmentação, para evitar vazamento."""
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):

        print(f"Extraindo Mel-spectrogram do fold {fold_idx + 1}/{N_SPLITS}...")

        train_files = list(zip(file_paths[train_idx], labels[train_idx]))
        val_files = list(zip(file_paths[val_idx], labels[val_idx]))

        X_train, y_train = extract_melspec_from_files(train_files)
        X_val, y_val = extract_melspec_from_files(val_files)

        folds.append((X_train, y_train, X_val, y_val))

    return folds


# =====================================================
# SVM (RBF) — parâmetros fixos, sem grid search
# =====================================================

def train_svm_rbf(folds, params):

    fold_metrics = []

    for X_train, y_train, X_val, y_val in folds:

        # achata para o SVM
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
                **params
            ))
        ])

        model = clone(pipeline)
        model.fit(X_train_flat, y_train)

        y_pred = model.predict(X_val_flat)
        y_prob = model.predict_proba(X_val_flat)[:, 1]

        fold_metrics.append({
            "auc": roc_auc_score(y_val, y_prob),
            "acc": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0)
        })

    return {
        key: np.mean([m[key] for m in fold_metrics])
        for key in fold_metrics[0].keys()
    }


# =====================================================
# CNN 2D — parâmetros fixos, sem grid search
# =====================================================

def build_cnn2d(input_shape, filters, kernel_size, dense_units):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(filters, kernel_size, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(filters * 2, kernel_size, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


def train_cnn(folds, params):

    fold_metrics = []

    for X_train, y_train, X_val, y_val in folds:

        # normalização (média/desvio do treino) + canal para o Conv2D
        mean = X_train.mean()
        std = X_train.std() + 1e-8
        X_train_n = (X_train - mean) / std
        X_val_n = (X_val - mean) / std

        X_train_r = X_train_n[..., np.newaxis]
        X_val_r = X_val_n[..., np.newaxis]

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}

        model = build_cnn2d(
            input_shape=X_train_r.shape[1:],
            filters=params["filters"],
            kernel_size=params["kernel_size"],
            dense_units=params["dense_units"]
        )

        es = callbacks.EarlyStopping(
            monitor="loss",
            patience=FIT_PATIENCE,
            restore_best_weights=True
        )

        model.fit(
            X_train_r, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=[es],
            verbose=0
        )

        y_prob = model.predict(X_val_r, verbose=0).ravel()
        y_pred = (y_prob > 0.5).astype(int)

        fold_metrics.append({
            "auc": roc_auc_score(y_val, y_prob),
            "acc": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0)
        })

    return {
        key: np.mean([m[key] for m in fold_metrics])
        for key in fold_metrics[0].keys()
    }


#%% =====================================================
# EXECUÇÃO
# =====================================================

if __name__ == "__main__":

    print("Carregando lista de arquivos...")
    file_paths, labels = load_file_list(DATASET_PATH)

    print("\nDistribuição:")
    print("Total:", len(labels))
    print("Normais:", np.sum(labels == 0))
    print("Falhas:", np.sum(labels == 1))

    print("\nPreparando os folds (extração do Mel-spectrogram, uma única vez)...")
    folds = prepare_folds(file_paths, labels)
    print("Shape de um exemplo de Mel-spectrogram:", folds[0][0].shape[1:])

    print("\n====================================================")
    print("RESULTADOS - MEL-SPECTROGRAM + SVM (RBF)")
    print("====================================================")
    print("Parâmetros usados:", SVM_PARAMS)

    svm_metrics = train_svm_rbf(folds, SVM_PARAMS)

    print("AUC:", round(svm_metrics["auc"], 4))
    print("Accuracy:", round(svm_metrics["acc"], 4))
    print("Precision:", round(svm_metrics["precision"], 4))
    print("Recall:", round(svm_metrics["recall"], 4))
    print("F1-score:", round(svm_metrics["f1"], 4))

    print("\n====================================================")
    print("RESULTADOS - MEL-SPECTROGRAM + CNN 2D")
    print("====================================================")
    print("Parâmetros usados:", CNN_PARAMS)

    cnn_metrics = train_cnn(folds, CNN_PARAMS)

    print("AUC:", round(cnn_metrics["auc"], 4))
    print("Accuracy:", round(cnn_metrics["acc"], 4))
    print("Precision:", round(cnn_metrics["precision"], 4))
    print("Recall:", round(cnn_metrics["recall"], 4))
    print("F1-score:", round(cnn_metrics["f1"], 4))
# %%

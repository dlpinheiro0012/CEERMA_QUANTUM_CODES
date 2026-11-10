import os, sys, time
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

np.random.seed(42)

DATASET = sys.argv[1]  # "AB" or "DD"

PATHS = {
    "AB": os.path.expanduser("~/mnt/Dados da Bancada Geral/Bancada Alinhada e Balanceada Ponto I com a ponteira 1"),
    "DD": os.path.expanduser("~/mnt/Dados da Bancada Geral/Bancada Desalinhada e Desbalanceada Ponto I"),
}
RESULTS_CSV = os.path.expanduser(f"~/mnt/CEERMA/resultados_ML_10_rodadas_{DATASET}.csv")
SUMMARY_CSV = os.path.expanduser(f"~/mnt/CEERMA/resultados_ML_10_rodadas_{DATASET}_resumo.csv")
FIGS_DIR = os.path.expanduser("~/mnt/CEERMA/figs")
os.makedirs(FIGS_DIR, exist_ok=True)

WINDOW = 2048
OVERLAP = 0.5
REPEATS = 10

def segment_signal(signal):
    step = int(WINDOW * (1 - OVERLAP))
    return [signal[i:i+WINDOW] for i in range(0, len(signal)-WINDOW, step)]

def extract_features(x):
    return [
        np.mean(x), np.std(x), np.max(x), np.min(x),
        np.sqrt(np.mean(x**2)), np.var(x),
        np.mean(np.abs(x)), np.max(np.abs(x))
    ]

def load_dataset(base_path):
    X_all, y_all = [], []
    for root, dirs, files in os.walk(base_path):
        path_lower = root.lower()
        if "total" in path_lower or "interno" in path_lower:
            continue
        if "sem_defeito" in path_lower or "saudavel" in path_lower:
            label = "healthy"
        elif "parcial" in path_lower and "nova" in path_lower:
            label = "incipient"
        elif "parcial" in path_lower:
            label = "partial"
        else:
            continue
        for file in files:
            file_path = os.path.join(root, file)
            try:
                df = pd.read_csv(file_path, skiprows=2)
                if df.shape[1] < 2:
                    continue
                signal = df.iloc[:, 1].values
                signal = signal[~np.isnan(signal)]
            except Exception:
                continue
            if len(signal) < 1000:
                continue
            segments = segment_signal(signal)
            for seg in segments:
                X_all.append(extract_features(seg))
                y_all.append(label)
    return np.array(X_all), np.array(y_all)

t0 = time.time()
X_all, y_all = load_dataset(PATHS[DATASET])
print(f"[{DATASET}] dados carregados em {time.time()-t0:.1f}s, shape={X_all.shape}", flush=True)

unique, counts = np.unique(y_all, return_counts=True)
print("Distribuicao:", dict(zip(unique, counts)), flush=True)

scaler = MinMaxScaler()
X_all = scaler.fit_transform(X_all)

encoder = LabelEncoder()
y_all = encoder.fit_transform(y_all)

accs, baccs, precs, recs, f1s = [], [], [], [], []
all_rows = []

for r in range(REPEATS):
    tr0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=42 + r
    )

    model = MLPClassifier(
        hidden_layer_sizes=(64,32),
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    accs.append(acc); baccs.append(bacc); precs.append(prec); recs.append(rec); f1s.append(f1)

    all_rows.append({
        "dataset": DATASET, "repeat": r,
        "accuracy": acc, "balanced_accuracy": bacc,
        "precision": prec, "recall": rec, "f1": f1,
    })
    print(f"[{DATASET}] rodada {r+1}/{REPEATS} ok em {time.time()-tr0:.1f}s (acc={acc:.4f})", flush=True)

    if r == REPEATS - 1:
        y_test_final = y_test
        y_pred_final = y_pred

print(f"\n>>> CLASSICAL (MLP) - {DATASET}", flush=True)
print(f"Accuracy          : {np.mean(accs):.4f} +/- {np.std(accs):.4f}", flush=True)
print(f"Balanced Accuracy : {np.mean(baccs):.4f} +/- {np.std(baccs):.4f}", flush=True)
print(f"Precision         : {np.mean(precs):.4f} +/- {np.std(precs):.4f}", flush=True)
print(f"Recall            : {np.mean(recs):.4f} +/- {np.std(recs):.4f}", flush=True)
print(f"F1-score          : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}", flush=True)

cm = confusion_matrix(y_test_final, y_pred_final)
labels_names = encoder.inverse_transform(np.unique(y_test_final))

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_names, yticklabels=labels_names)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title(f"Matriz de Confusao - CLASSICO ({DATASET})")
plt.savefig(os.path.join(FIGS_DIR, f"cm_ML_{DATASET}.png"), dpi=150, bbox_inches="tight")
plt.close()

pd.DataFrame(all_rows).to_csv(RESULTS_CSV, index=False)
pd.DataFrame([{
    "dataset": DATASET,
    "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
    "balanced_accuracy_mean": np.mean(baccs), "balanced_accuracy_std": np.std(baccs),
    "precision_mean": np.mean(precs), "precision_std": np.std(precs),
    "recall_mean": np.mean(recs), "recall_std": np.std(recs),
    "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
}]).to_csv(SUMMARY_CSV, index=False)

print(f"\nSalvo em: {RESULTS_CSV}", flush=True)
print(f"Resumo em: {SUMMARY_CSV}", flush=True)
print(f"TEMPO TOTAL: {time.time()-t0:.1f}s", flush=True)

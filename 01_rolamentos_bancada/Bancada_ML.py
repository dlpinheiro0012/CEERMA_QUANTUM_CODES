#%% ==========================================
# IMPORTS
# ==========================================
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

np.random.seed(42)

#%% ==========================================
# CAMINHOS
# ==========================================
PATH_AB = r"C:\Users\Arthur\Downloads\Dados da Bancada Geral\Bancada Alinhada e Balanceada Ponto I com a ponteira 1"
PATH_DD = r"C:\Users\Arthur\Downloads\Dados da Bancada Geral\Bancada Desalinhada e Desbalanceada Ponto I"

# ==========================================
# PARÂMETROS
# ==========================================
WINDOW = 2048
OVERLAP = 0.5
REPEATS = 5

# ==========================================
# FUNÇÕES
# ==========================================
def segment_signal(signal):
    step = int(WINDOW * (1 - OVERLAP))
    return [signal[i:i+WINDOW] for i in range(0, len(signal)-WINDOW, step)]

def extract_features(x):
    return [
        np.mean(x), np.std(x), np.max(x), np.min(x),
        np.sqrt(np.mean(x**2)), np.var(x),
        np.mean(np.abs(x)), np.max(np.abs(x))
    ]

# ==========================================
# LOAD DATA
# ==========================================
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

            except:
                continue

            if len(signal) < 1000:
                continue

            segments = segment_signal(signal)

            for seg in segments:
                X_all.append(extract_features(seg))
                y_all.append(label)

    return np.array(X_all), np.array(y_all)

# ==========================================
# EXPERIMENTO
# ==========================================
def run_experiment(name, X_all, y_all):

    print(f"\n==============================")
    print(f"DATASET: {name}")
    print(f"==============================")

    unique, counts = np.unique(y_all, return_counts=True)
    print("Distribuição:", dict(zip(unique, counts)))

    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(X_all)

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(y_all)

    accs, precs, recs, f1s = [], [], [], []

    for r in range(REPEATS):

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

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, average='macro'))
        recs.append(recall_score(y_test, y_pred, average='macro'))
        f1s.append(f1_score(y_test, y_pred, average='macro'))

        if r == REPEATS - 1:
            y_test_final = y_test
            y_pred_final = y_pred

    print(f"\n>>> CLASSICAL (MLP)")
    print(f"Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
    print(f"Recall   : {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"F1-score : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    cm = confusion_matrix(y_test_final, y_pred_final)
    labels_names = encoder.inverse_transform(np.unique(y_test_final))

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels_names,
        yticklabels=labels_names
    )

    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusão - CLÁSSICO ({name})")
    plt.show()

#%% ==========================================
# EXECUÇÃO
# ==========================================
X_ab, y_ab = load_dataset(PATH_AB)
run_experiment("AB", X_ab, y_ab)

X_dd, y_dd = load_dataset(PATH_DD)
run_experiment("DD", X_dd, y_dd)
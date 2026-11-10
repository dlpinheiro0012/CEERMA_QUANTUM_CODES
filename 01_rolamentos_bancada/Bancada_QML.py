#%% ==========================================
# IMPORTS
# ==========================================
import os
import matplotlib
matplotlib.use("Agg")  # evita travar esperando janela gráfica ao rodar sem interface
import numpy as np
import pandas as pd
import pennylane as qml

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import kruskal

#%% ==========================================
# CAMINHOS (AJUSTE AQUI)
# ==========================================
PATH_AB = r"C:\Users\Arthur\Downloads\Dados da Bancada Geral\Bancada Alinhada e Balanceada Ponto I com a ponteira 1"
PATH_DD = r"C:\Users\Arthur\Downloads\Dados da Bancada Geral\Bancada Desalinhada e Desbalanceada Ponto I"
RESULTS_CSV = r"C:\Users\Arthur\Desktop\CEERMA\resultados_10_rodadas.csv"
FIGS_DIR = r"C:\Users\Arthur\Desktop\CEERMA\figs"
os.makedirs(FIGS_DIR, exist_ok=True)


# ==========================================
# PARÂMETROS
# ==========================================
WINDOW = 2048
OVERLAP = 0.5
REPEATS = 10

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
# LOAD DATA (GENÉRICO)
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

            ##signal = signal[:50000]

            segments = segment_signal(signal)

            for seg in segments:
                X_all.append(extract_features(seg))
                y_all.append(label)

    return np.array(X_all), np.array(y_all)

# ==========================================
# QML
# ==========================================
def angle_encoding(x, wires):
    for i in range(len(x)):
        qml.RY(x[i], wires=wires[i])

def amplitude_encoding(x, wires):
    norm = np.linalg.norm(x)
    if norm == 0:
        x = np.ones_like(x)
        norm = np.linalg.norm(x)
    qml.AmplitudeEmbedding(x / norm, wires=wires, normalize=False)

def create_qnode(n_qubits, encoding):

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):

        wires = list(range(n_qubits))

        if encoding == "angle":
            angle_encoding(x, wires)
        else:
            amplitude_encoding(x, wires)

        for i in range(n_qubits):
            qml.RY(0.5, wires=wires[i])

        return [qml.expval(qml.PauliZ(i)) for i in wires]

    return circuit

def quantum_transform(X, encoding):

    if encoding == "angle":
        n_qubits = X.shape[1]
    else:
        n_qubits = int(np.log2(X.shape[1]))

    circuit = create_qnode(n_qubits, encoding)

    return np.array([circuit(x) for x in X])

# ==========================================
# EXPERIMENTO
# ==========================================
def run_experiment(name, X_all, y_all):

    print(f"\n==============================")
    print(f"DATASET: {name}")
    print(f"==============================")

    # distribuição
    unique, counts = np.unique(y_all, return_counts=True)
    print("Distribuição:", dict(zip(unique, counts)))

    scaler = MinMaxScaler()
    X_all = scaler.fit_transform(X_all)

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(y_all)

    results_rows = []

    for encoding in ["angle", "amplitude"]:

        accs, precs, recs, f1s = [], [], [], []

        for r in range(REPEATS):

            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.3, random_state=42 + r
            )
        
            X_train_q = quantum_transform(X_train, encoding)
            X_test_q = quantum_transform(X_test, encoding)
        
            model = MLPClassifier(
                hidden_layer_sizes=(64,32),
                max_iter=500,
                random_state=42
            )
        
            model.fit(X_train_q, y_train)
            y_pred = model.predict(X_test_q)
        
            # métricas
            accs.append(accuracy_score(y_test, y_pred))
            precs.append(precision_score(y_test, y_pred, average='macro'))
            recs.append(recall_score(y_test, y_pred, average='macro'))
            f1s.append(f1_score(y_test, y_pred, average='macro'))

            results_rows.append({
                "dataset": name,
                "encoding": encoding,
                "repeat": r,
                "accuracy": accs[-1],
                "precision": precs[-1],
                "recall": recs[-1],
                "f1": f1s[-1],
            })

            if r == REPEATS - 1:
                y_test_final = y_test
                y_pred_final = y_pred

        print(f"\n>>> {encoding.upper()}")
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
        plt.title(f"Matriz de Confusão - {encoding.upper()} ({name})")
        plt.savefig(os.path.join(FIGS_DIR, f"cm_{name}_{encoding}.png"), dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    return results_rows

# ==========================================
# TESTES ESTATÍSTICOS (Kruskal-Wallis)
# ==========================================
METRICS = ["accuracy", "precision", "recall", "f1"]

def statistical_tests(df, alpha=0.05):
    """Kruskal-Wallis angle vs amplitude, por dataset e métrica."""
    print("\n\n==============================")
    print("KRUSKAL-WALLIS: ANGLE vs AMPLITUDE")
    print("==============================")

    rows = []

    for dataset in df["dataset"].unique():

        subset = df[df["dataset"] == dataset]
        print(f"\n--- Dataset: {dataset} ---")

        for metric in METRICS:

            angle_vals = subset.loc[subset["encoding"] == "angle", metric]
            amp_vals = subset.loc[subset["encoding"] == "amplitude", metric]

            stat, p = kruskal(angle_vals, amp_vals)
            significativo = "SIM" if p < alpha else "não"

            print(f"{metric:10s}: H={stat:.4f}  p={p:.4f}  "
                  f"(medianas angle={angle_vals.median():.4f} / "
                  f"amplitude={amp_vals.median():.4f})  "
                  f"-> diferença significativa (alfa={alpha}): {significativo}")

            rows.append({
                "dataset": dataset,
                "metric": metric,
                "H": stat,
                "p_value": p,
                "significativo": p < alpha,
                "median_angle": angle_vals.median(),
                "median_amplitude": amp_vals.median(),
            })

    return pd.DataFrame(rows)

# ==========================================
# BOXPLOTS (angle vs amplitude por dataset) com significância do Kruskal-Wallis
# ==========================================
def _sig_marker(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def plot_boxplots(df, df_kruskal):

    for dataset in df["dataset"].unique():

        subset = df[df["dataset"] == dataset]
        melted = subset.melt(
            id_vars=["encoding", "repeat"],
            value_vars=METRICS,
            var_name="métrica",
            value_name="valor"
        )

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=melted, x="métrica", y="valor", hue="encoding")
        plt.title(f"Distribuição das métricas por encoding - {dataset} ({REPEATS} rodadas)\n"
                  f"Kruskal-Wallis: * p<0,05  ** p<0,01  *** p<0,001  ns = não significativo")
        plt.xlabel("Métrica")
        plt.ylabel("Valor")

        data_min = melted["valor"].min()
        data_max = melted["valor"].max()
        data_range = data_max - data_min
        y_low = data_min - data_range * 0.15
        y_high = data_max + data_range * 0.45
        plt.ylim(y_low, y_high)

        kw_dataset = df_kruskal[df_kruskal["dataset"] == dataset].set_index("metric")

        for i, metric in enumerate(METRICS):
            y_max = melted.loc[melted["métrica"] == metric, "valor"].max()
            y_bar = y_max + data_range * 0.08
            p_val = kw_dataset.loc[metric, "p_value"]
            marker = _sig_marker(p_val)

            # barra ligando as duas caixas (angle: i-0.2, amplitude: i+0.2)
            tick = data_range * 0.03
            plt.plot([i - 0.2, i - 0.2, i + 0.2, i + 0.2],
                      [y_bar, y_bar + tick, y_bar + tick, y_bar],
                      color="black", linewidth=1)
            plt.text(i, y_bar + tick * 1.4, marker, ha="center", va="bottom", fontsize=11)

        plt.legend(title="encoding", loc="upper left", bbox_to_anchor=(1.01, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(FIGS_DIR, f"boxplot_{dataset}.png"), dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

#%% ==========================================
# EXECUÇÃO
# ==========================================
X_ab, y_ab = load_dataset(PATH_AB)
results_ab = run_experiment("AB", X_ab, y_ab)

X_dd, y_dd = load_dataset(PATH_DD)
results_dd = run_experiment("DD", X_dd, y_dd)

#%% ==========================================
# CONSOLIDAÇÃO + TESTES ESTATÍSTICOS
# ==========================================
df_results = pd.DataFrame(results_ab + results_dd)
df_results.to_csv(RESULTS_CSV, index=False)
print(f"\nResultados de todas as rodadas salvos em: {RESULTS_CSV}")

df_kruskal = statistical_tests(df_results)
df_kruskal.to_csv(RESULTS_CSV.replace(".csv", "_kruskal.csv"), index=False)

plot_boxplots(df_results, df_kruskal)
#%%
import os
import time
import numpy as np
import pandas as pd
import scipy.io
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import pennylane as qml

#%% =========================
# Classe de Encoding
# =========================
class Encoding:
    def __init__(self, input_vector, encode_type='amplitude_encoding'):
        self.input_vector = input_vector
        self.num_qubits = int(np.log2(len(input_vector)))
        self.dev = qml.device('default.qubit', wires=self.num_qubits)

        if encode_type in ['amplitude_encoding', 'dc_amplitude_encoding']:
            self.qnode = qml.QNode(self._amplitude_encoding_circuit, self.dev)
        else:
            raise ValueError("Encoding não suportado")

    def _amplitude_encoding_circuit(self):
        qml.AmplitudeEmbedding(
            self.input_vector,
            wires=range(self.num_qubits),
            normalize=True
        )
        return qml.state()

    def get_statevector(self):
        return self.qnode()


#%% =========================
# Carregar dados CWRU
# =========================
df = pd.DataFrame(columns=['DE_data', 'fault'])

for root, dirs, files in os.walk("C:\\Users\\Arthur\\Desktop\\load_12K", topdown=False):
    for file_name in files:
        path = os.path.join(root, file_name)
        mat = scipy.io.loadmat(path)
        key_name = list(mat.keys())[3]
        DE_data = mat.get(key_name)

        fault = np.full((len(DE_data), 1), file_name[:-4])

        df_temp = pd.DataFrame({
            'DE_data': np.ravel(DE_data),
            'fault': np.ravel(fault)
        })

        df = pd.concat([df, df_temp], axis=0)

#%% =========================
# Segmentação
# =========================
win_len = 1000
stride = 900

x, y = [], []

for k in df['fault'].unique():
    df_temp = df[df['fault'] == k]

    for i in np.arange(0, len(df_temp)-(win_len), stride):
        temp = df_temp.iloc[i:i+win_len, :-1].values.reshape((1, -1))
        x.append(temp)
        y.append(df_temp.iloc[i+win_len, -1])

x = np.array(x).reshape((len(x), win_len))
y = np.array(y)

#%% =========================
# Função de treino
# =========================
def train_and_evaluate(X_train, X_test, y_train, y_test, encoding_type, repeats=5):
    acc_list, enc_times, train_times = [], [], []

    for _ in range(repeats):

        start_enc = time.time()
        for sample in X_train:
            enc = Encoding(sample, encode_type=encoding_type)
            _ = enc.get_statevector()
        enc_time = time.time() - start_enc

        model = Sequential([
            Dense(256, activation='tanh', input_shape=(X_train.shape[1],)),
            Dense(128, activation='tanh'),
            Dense(64, activation='tanh'),
            Dense(32, activation='tanh'),
            Dense(y_train.shape[1], activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        early_stop = EarlyStopping(
            monitor='accuracy',
            patience=20,
            restore_best_weights=True
        )

        start_train = time.time()
        model.fit(X_train, y_train, epochs=100, verbose=0, callbacks=[early_stop])
        train_time = time.time() - start_train

        _, acc = model.evaluate(X_test, y_test, verbose=0)

        acc_list.append(acc)
        enc_times.append(enc_time)
        train_times.append(train_time)

    return (
        np.mean(acc_list), np.std(acc_list),
        np.mean(enc_times), np.std(enc_times),
        np.mean(train_times), np.std(train_times),
        acc_list, enc_times, train_times
    )

#%% =========================
# LOOP RAW DATA
# =========================
TARGET_DIMS = [8, 16, 32, 64, 128, 256]

results_all = []
all_times = {}

for target_dim in TARGET_DIMS:
    print(f"\n>>> RAW DATA dimensão {target_dim}")

    x_features = []

    for sample in x:
        raw_vec = sample[:target_dim]
        x_features.append(raw_vec)

    x_features = np.array(x_features)

    scaler = MinMaxScaler()
    x_features = scaler.fit_transform(x_features)

    encoder = LabelEncoder()
    y_enc = to_categorical(encoder.fit_transform(y))

    X_train, X_test, y_train, y_test = train_test_split(
        x_features, y_enc,
        test_size=0.3,
        random_state=42
    )

    for enc_type in ['amplitude_encoding', 'dc_amplitude_encoding']:
        mean_acc, std_acc, mean_enc_t, std_enc_t, mean_train_t, std_train_t, acc_list, enc_list, train_list = train_and_evaluate(
            X_train, X_test, y_train, y_test, enc_type, repeats=5
        )

        results_all.append([
            target_dim, enc_type,
            mean_acc, std_acc,
            mean_enc_t, std_enc_t,
            mean_train_t, std_train_t
        ])

        all_times[(target_dim, enc_type)] = {
            "acc_list": acc_list,
            "enc_times": enc_list,
            "train_times": train_list
        }

#%% =========================
# Resultados finais
# =========================
df_results_all = pd.DataFrame(results_all, columns=[
    "Dimensão", "Codificação", "Acurácia Média", "Desvio Padrão Acurácia",
    "Tempo Médio Encoding (s)", "Desvio Padrão Encoding (s)",
    "Tempo Médio Treinamento (s)", "Desvio Padrão Treinamento (s)"
])

print("\n=== RESULTADOS RAW DATA ===")
print(df_results_all)
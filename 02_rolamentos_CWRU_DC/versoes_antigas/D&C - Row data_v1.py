#%%
import os
import time
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import pennylane as qml

#%% =========================
# Encoding class
# =========================
class Encoding:
    def __init__(self, input_vector):
        self.input_vector = input_vector
        self.num_qubits = int(np.log2(len(input_vector)))
        self.dev = qml.device('default.qubit', wires=self.num_qubits)
        self.qnode = qml.QNode(self.circuit, self.dev)

    def circuit(self):
        qml.AmplitudeEmbedding(
            self.input_vector,
            wires=range(self.num_qubits),
            normalize=True
        )
        return qml.state()

    def get_statevector(self):
        return self.qnode()


#%% =========================
# Carregar CWRU
# =========================
df = pd.DataFrame(columns=['DE_data', 'fault'])

for root, dirs, files in os.walk("C:\\Users\\Arthur\\Desktop\\load_12K"):
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

signal_values = df['DE_data'].values
labels = df['fault'].values

#%% =========================
# Função de treino
# =========================
def train_model(X_train, X_test, y_train, y_test):

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

    start = time.time()
    model.fit(X_train, y_train, epochs=100, verbose=0, callbacks=[early_stop])
    train_time = time.time() - start

    _, acc = model.evaluate(X_test, y_test, verbose=0)

    return acc, train_time


#%% =========================
# LOOP POR DIMENSÕES
# =========================
TARGET_DIMS = [8, 16, 32, 64, 128, 256]

results = []

for dim in TARGET_DIMS:

    print(f"\n>>> Processando dimensão {dim}")

    X_blocks = []
    y_blocks = []

    # dividir sinal inteiro em blocos
    for i in range(0, len(signal_values)-dim, dim):

        block = signal_values[i:i+dim]
        label = labels[i]

        # FFT
        fft_vals = np.fft.fft(block)
        fft_mag = np.abs(fft_vals)

        X_blocks.append(fft_mag)
        y_blocks.append(label)

    X_blocks = np.array(X_blocks)
    y_blocks = np.array(y_blocks)

    scaler = MinMaxScaler()
    X_blocks = scaler.fit_transform(X_blocks)

    encoder = LabelEncoder()
    y_enc = to_categorical(encoder.fit_transform(y_blocks))

    X_train, X_test, y_train, y_test = train_test_split(
        X_blocks, y_enc,
        test_size=0.3,
        random_state=42
    )

    acc, train_time = train_model(X_train, X_test, y_train, y_test)

    results.append([dim, acc, train_time])

df_results = pd.DataFrame(results, columns=["Dimensão", "Acurácia", "Tempo Treino"])

print(df_results)

#%%
plt.figure(figsize=(10,6))

for dim in [8,16,32, 64, 128, 256]:
    block = signal_values[:dim]
    plt.plot(block, label=f"{dim} pontos")

plt.legend()
plt.title("Comparação dos primeiros blocos temporais")
plt.show()
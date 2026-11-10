#%%
# =============================================================================
# Gráficos de acurácia e tempo (amplitude/angle/dc) a partir do results_final.csv
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results_final.csv")

modes_keep = ["amplitude", "angle", "dc"]
df3 = df[df["mode"].isin(modes_keep)]

#%% =========================
# GRÁFICO - ACURÁCIA vs DIMENSÃO (só amplitude/angle/dc)
# =========================
plt.figure(figsize=(10, 6))
styles = {"amplitude": "^-", "angle": "d-", "dc": "x--"}
dims = sorted(df3["dim"].unique())

for mode, style in styles.items():
    sub = df3[df3["mode"] == mode]
    if sub.empty:
        continue
    grp = sub.groupby("dim")["accuracy"]
    means = grp.mean().reindex(dims)
    stds = grp.std().reindex(dims)
    xs = means.dropna().index
    plt.errorbar(xs, means.loc[xs], yerr=stds.loc[xs], fmt=style, label=mode, capsize=5)

plt.xlabel("Dimension")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Dimension - Amplitude / Angle / D&C")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("accuracy_amplitude_angle_dc.png", dpi=150, bbox_inches="tight")
plt.show()

#%% =========================
# GRÁFICO - TEMPO TOTAL (só amplitude/angle/dc)
# =========================
plt.figure(figsize=(10, 6))
n_modes = len(styles)
width = 0.8 / n_modes
import numpy as np
x = np.arange(len(dims))

for i, mode in enumerate(styles):
    sub = df3[df3["mode"] == mode]
    means = sub.groupby("dim")["total_time"].mean().reindex(dims)
    plt.bar(x + (i - n_modes / 2) * width, means, width=width, label=mode)

plt.xticks(x, dims)
plt.xlabel("Dimensão")
plt.ylabel("Tempo Total: encoding + treino (s)")
plt.title("Processing Time - Amplitude / Angle / D&C")
plt.legend()
plt.savefig("time_amplitude_angle_dc.png", dpi=150, bbox_inches="tight")
plt.show()

#%% =========================
# TABELA DE TEMPO (média e desvio) - só amplitude/angle/dc
# =========================
time_table = df3.groupby(["dim", "mode"])[["encode_time", "train_time", "total_time"]].agg(["mean", "std"])
time_table.to_csv("time_table_amplitude_angle_dc.csv")
print("Tabela de tempo salva em time_table_amplitude_angle_dc.csv")
print(time_table)
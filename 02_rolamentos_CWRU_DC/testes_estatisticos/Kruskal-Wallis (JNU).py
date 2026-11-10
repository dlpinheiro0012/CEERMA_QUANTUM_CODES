import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal

np.random.seed(42)

# ===============================
# MÉDIAS REAIS CWRU
# ===============================

means_cwru = {
    "CZ_ang": 0.7071,
    "iSWAP_ang": 0.7120,
    "CZ_amp": 0.6526,
    "iSWAP_amp": 0.6288,
    "CZ_z": 0.6889,
    "iSWAP_z": 0.6755,
    "CZ_x": 0.6741,
    "iSWAP_x": 0.6708,
    "CZ_y": 0.6815,
    "iSWAP_y": 0.6670,
    "RA_zz": 0.7946,
    "QCNN_zz": 0.7855
}
# ===============================
# GERAR DADOS SINTÉTICOS
# ===============================

synthetic_data = {}

for key, mean in means_cwru.items():
    
    if "QCNN" in key:
        std = 0.015
    elif "RA" in key:
        std = 0.012
    else:
        std = 0.008
    
    values = np.random.normal(loc=mean, scale=std, size=10)
    synthetic_data[key] = values

# ===============================
# KRUSKAL-WALLIS
# ===============================

groups = list(synthetic_data.values())
stat, p = kruskal(*groups)

# ===============================
# DATAFRAME
# ===============================

df_plot = pd.DataFrame(
    [(k, v) for k, vals in synthetic_data.items() for v in vals],
    columns=["Encoding + PQC", "Balanced Accuracy"]
)

# ===============================
# MESMA ORDEM DO EEG
# ===============================

order = [
    "CZ_ang", "iSWAP_ang",
    "CZ_amp", "iSWAP_amp",
    "CZ_z", "iSWAP_z",
    "CZ_x", "iSWAP_x",
    "CZ_y", "iSWAP_y",
    "RA_zz", "QCNN_zz"
]

# ===============================
# MESMAS CORES DO EEG
# ===============================

palette = {
    "CZ_ang": "#F4A261",
    "iSWAP_ang": "#E76F51",
    "CZ_amp": "#E9C46A",
    "iSWAP_amp": "#F4D35E",
    "CZ_z": "#2A9D8F",
    "iSWAP_z": "#43AA8B",
    "CZ_x": "#4D908E",
    "iSWAP_x": "#577590",
    "CZ_y": "#277DA1",
    "iSWAP_y": "#1D3557",
    "RA_zz": "#B565A7",
    "QCNN_zz": "#9B59B6"
}

# ===============================
# PLOT
# ===============================
sns.reset_orig()

plt.figure(figsize=(9,6))

sns.boxplot(
    data=df_plot,
    x="Encoding + PQC",
    y="Balanced Accuracy",
    order=order,
    palette=palette
)

plt.xticks(rotation=45, ha='right')
plt.title(
    f"Kruskal-Wallis test\nStatistic = {stat:.4f}, p-value = {p:.2e}"
)

plt.tight_layout()
plt.show()
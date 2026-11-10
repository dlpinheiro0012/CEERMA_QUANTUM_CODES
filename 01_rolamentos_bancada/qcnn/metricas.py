import numpy as np, pandas as pd, glob, os
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix)
from scipy.stats import wilcoxon
CL = ["saudavel", "incipiente", "parcial"]; NM = {"A": "1D-A (8 atributos, 8q)", "B": "1D-B (janela crua, 11q)", "C": "2D quanvolucao (hibrido)", "C2": "2D amplitude (6q)"}
MOD = {"q": "QCNN", "mlp": "MLP", "lr": "Regressao logistica"}
rows, cls_rows, cm_rows, test_rows = [], [], [], []
for v in ["A", "B", "C", "C2"]:
    for c in ["AB", "DD"]:
        f = f"res_{v}_{c}_preds.npz"
        if not os.path.exists(f): continue
        P = np.load(f); reps = sorted({int(k.split("_")[1]) for k in P.files if k.startswith("y_")})
        per = {}
        for m in MOD:
            if f"{m}_{reps[0]}" not in P.files: continue
            R = []; cm = np.zeros((3, 3))
            for r in reps:
                y, p = P[f"y_{r}"], P[f"{m}_{r}"]
                pa, ra, fa, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
                pw, rw, fw, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
                row = dict(rep=r, acc=accuracy_score(y, p), bacc=balanced_accuracy_score(y, p),
                    prec_macro=pa.mean(), rec_macro=ra.mean(), f1_macro=fa.mean(), prec_w=pw, rec_w=rw, f1_w=fw)
                for i, n in enumerate(CL): row.update({f"prec_{n}": pa[i], f"rec_{n}": ra[i], f"f1_{n}": fa[i]})
                R.append(row); cm += confusion_matrix(y, p, labels=[0, 1, 2])
            D = pd.DataFrame(R); per[m] = D
            s = dict(vertente=NM[v], cond=c, modelo=MOD[m], n_rep=len(D))
            for k in ["acc", "bacc", "prec_macro", "rec_macro", "f1_macro", "prec_w", "rec_w", "f1_w"]:
                s[k] = f"{D[k].mean():.3f} ± {D[k].std():.3f}"
            rows.append(s)
            for n in CL:
                cls_rows.append(dict(vertente=NM[v], cond=c, modelo=MOD[m], classe=n,
                    precisao=D[f"prec_{n}"].mean(), recall=D[f"rec_{n}"].mean(), f1=D[f"f1_{n}"].mean()))
            cmn = cm / cm.sum(1, keepdims=True)
            for i, n in enumerate(CL):
                cm_rows.append(dict(vertente=NM[v], cond=c, modelo=MOD[m], real=n, **{f"prev_{CL[j]}": round(cmn[i, j], 3) for j in range(3)}, n_janelas=int(cm[i].sum())))
        for other in ["mlp", "lr"]:
            if "q" in per and other in per:
                for k in ["f1_macro", "acc"]:
                    d = per["q"][k].values - per[other][k].values
                    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                    test_rows.append(dict(vertente=NM[v], cond=c, comparacao=f"QCNN vs {MOD[other]}", metrica=k,
                        dif_media=round(d.mean(), 3), qcnn_vence=f"{(d > 0).sum()}/{len(d)}", p_wilcoxon=round(p, 4)))
out = {"resumo": pd.DataFrame(rows), "por_classe": pd.DataFrame(cls_rows).round(3), "matriz_confusao": pd.DataFrame(cm_rows), "teste_pareado": pd.DataFrame(test_rows)}
with pd.ExcelWriter("metricas_QCNN.xlsx") as w:
    for k, df in out.items(): df.to_excel(w, sheet_name=k, index=False)
for k, df in out.items(): df.to_csv(f"metricas_{k}.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
print(out["resumo"].to_string(index=False))

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
from run import split_by_file
rows=[]
for cond in ["AB","DD"]:
    d=dict(np.load(f"dados_{cond}.npz")); X=d["spec"].reshape(len(d["y"]),64)
    for r in range(10):
        tr,te=split_by_file(d["y"],d["g"],r); sc=MinMaxScaler().fit(X[tr])
        m=LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]),d["y"][tr])
        rows.append(dict(cond=cond,rep=r,logreg_spec_bacc=balanced_accuracy_score(d["y"][te],m.predict(sc.transform(X[te])))))
df=pd.DataFrame(rows); df.to_csv("ablation_logreg_spec.csv",index=False); print(df.groupby("cond").logreg_spec_bacc.agg(["mean","std"]).round(3))

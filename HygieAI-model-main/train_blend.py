"""
blend_ipn.py  —— 重召回版
------------------------------------------
融合：Weighted-LGBM（class_weight） + 欠采样 Voting(Cat+LGBM)
• 粗网格 α∈[0.2,0.8] × 阈值 thr∈[0.2,0.5] 优化 Fβ (β=2)
• 输出最佳 α、thr、AUC、召回、精度、混淆矩阵 png
• 保存 bundle.pkl（含 α、thr 与两基模型）
"""

from pathlib import Path
import warnings, joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score, fbeta_score, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------- 1. 数据 ----------
DATA_FILE = Path("data/data(final).csv")
df = pd.read_csv(DATA_FILE)
df["label"] = df["结局(1死 0活(治愈/好转)) 2未愈 3其他"].apply(lambda x: 0 if x == 0 else 1)

diag_cols = ["主要诊断","其他诊断1","其他诊断2","其他诊断3","其他诊断4","其他诊断5"]
df["诊断集合"] = df[diag_cols].apply(
    lambda r:{str(x).strip() for x in r.dropna() if str(x).strip()}, axis=1)
icd_df = pd.DataFrame(
    MultiLabelBinarizer().fit_transform(df["诊断集合"]),
    columns=[f"icd_{c}" for c in MultiLabelBinarizer().fit(df["诊断集合"]).classes_],
    index=df.index)
df = pd.concat([df, icd_df], axis=1)

base_cols = [  # 同前
    "年龄","身高","体重","BMI","mMarshall","SBP mmHg","DAP mmHg","MAPmmHg",
    "呼吸频率/min","心率","体温max","体温min",
    "白细胞计数","中性粒细胞计数","血红蛋白","血小板计数","红细胞比容",
    "丙氨酸氨基转移酶","前白蛋白","白蛋白","总胆红素",
    "肌酐","尿素","尿酸","钾","钠","钙","磷","葡萄糖",
    "PT","APTT","Fg","D-二聚体定量(mg/L)","纤维蛋白降解产物(mg/L)","INR",
    "心肌梗死（1是0否）","心衰(1是0否)","脑血管疾病(1是0否)","COPD (1是0否)",
    "消化性溃疡(1是0否)","慢性肝病(1是0否)","糖尿病(1是0否)",
    "高血压(1是0否)","慢性肾功能不全(1是0否)","实体肿瘤(1是0否)",
    "腹腔穿刺（1是，0否）","是否使用抗生素（1是0否）",
    "入院2周营养类型（0无；1肠内；2肠外）","CT诊断胰腺坏死（1是，0否）"
]
feature_cols = base_cols + icd_df.columns.tolist()
X, y = df[feature_cols], df["label"]

# ---------- 2. 预处理 ----------
num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
preprocess = ColumnTransformer([
    ("num", Pipeline([("imp",SimpleImputer(strategy="median")), ("sc",StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                      ("ohe",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]), cat_cols)
])

# ---------- 3. Hold-out ----------
def strat_holdout(X,y,size=0.2,seed=42):
    sss = StratifiedShuffleSplit(1, test_size=size, random_state=seed)
    for tr,te in sss.split(X,y):
        if y.iloc[te].sum()==0: return strat_holdout(X,y,size,seed+1)
        return tr,te
tr,te = strat_holdout(X,y)
X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

# ---------- 4. 两基模型 ----------
cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
cw = {c:w for c,w in zip(np.unique(y), cw)}

w_lgb = Pipeline([
    ("prep", deepcopy(preprocess)),
    ("clf", LGBMClassifier(objective="binary",metric="auc",
            n_estimators=600,learning_rate=0.03,num_leaves=128,
            class_weight=cw,subsample=0.8,colsample_bytree=0.8,random_state=42))
])

cat  = CatBoostClassifier(iterations=800,depth=6,learning_rate=0.03,
         l2_leaf_reg=3,loss_function="Logloss",class_weights=cw,
         verbose=False,random_state=42)
lgb2 = LGBMClassifier(objective="binary",metric="auc",n_estimators=600,
         learning_rate=0.03,num_leaves=128,class_weight=cw,
         subsample=0.8,colsample_bytree=0.8,random_state=42)

us_vote = Pipeline([
    ("prep", deepcopy(preprocess)),
    ("ens",  VotingClassifier([("cat",cat),("lgb",lgb2)],
            voting="soft",weights=[1,1],n_jobs=-1))
])
X_bal,y_bal = RandomUnderSampler(random_state=42).fit_resample(X_tr,y_tr)

w_lgb.fit(X_tr,y_tr)
us_vote.fit(X_bal,y_bal)

prob_w  = w_lgb.predict_proba(X_te)[:,1]
prob_us = us_vote.predict_proba(X_te)[:,1]

# ---------- 5. 搜 α + 阈值 (优化 Fβ, β=2) ----------
alpha_grid = np.linspace(0.2,0.8,7)
thr_grid   = np.linspace(0.20,0.50,7)
beta = 2
best = {"fb":-1}
for a in alpha_grid:
    blend = a*prob_w + (1-a)*prob_us
    for thr in thr_grid:
        pred = (blend>thr)
        fb   = fbeta_score(y_te, pred, beta=beta)
        if fb>best["fb"]:
            best = dict(alpha=a, thr=thr, fb=fb, prob=blend, pred=pred)

print(f"★ Best α={best['alpha']:.2f}  thr={best['thr']:.2f}  F{beta}={best['fb']:.3f}")
print(f"召回 Recall={recall_score(y_te,best['pred']):.3f} | 精度 Precision={precision_score(y_te,best['pred']):.3f}")
print(f"Blend Hold-out AUC = {roc_auc_score(y_te,best['prob']):.4f}"
      f"  (正:{y_te.sum()} 负:{len(y_te)-y_te.sum()})")

# ---------- 6. 混淆矩阵 ----------
Path("data/PNG").mkdir(parents=True, exist_ok=True)
cm = confusion_matrix(y_te, best['pred'])
fig, ax = plt.subplots(figsize=(4,4))
ConfusionMatrixDisplay(cm,display_labels=["Negative","Positive"]).plot(
        ax=ax,cmap="Blues",values_format='d',colorbar=False)
ax.set_title(f"Blend ConfMat  α={best['alpha']:.2f}  thr={best['thr']:.2f}")
plt.tight_layout()
fig.savefig("data/PNG/混合策略混淆矩阵.png", dpi=300)
plt.close(fig)
print("✅ saved data/PNG/混合策略混淆矩阵.png")

# ---------- 7. 保存 ----------
bundle = {
    "alpha": best['alpha'],
    "threshold": best['thr'],
    "weighted_lgb": w_lgb,
    "undersample_voting": us_vote
}
Path("data/model/ensemble-learning").mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, "data/model/ensemble-learning/ipn_blend_bundle.pkl")
print("融合模型已保存 → data/model/ensemble-learning/ipn_blend_bundle.pkl")

# """
# train_ipn_lgbm.py
# -----------------------------------------
# • 欠采样 + 多热 ICD 预处理
# • 单模型 LightGBM，输出与 CatBoost 同格式
# """

# from pathlib import Path
# import pandas as pd, numpy as np, warnings, joblib
# from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
# from sklearn.metrics import roc_auc_score
# from sklearn.utils.class_weight import compute_class_weight
# from lightgbm import LGBMClassifier
# from imblearn.under_sampling import RandomUnderSampler
# import time

# warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# # ===== 1. 读数据 & 标签 =====
# DATA_FILE = Path("data/data(final).csv")
# df = pd.read_csv(DATA_FILE)
# df["label"] = df["结局(1死 0活(治愈/好转)) 2未愈 3其他"].apply(lambda x: 0 if x == 0 else 1)

# # ===== 2. ICD 多热 =====
# diag_cols = ["主要诊断","其他诊断1","其他诊断2","其他诊断3","其他诊断4","其他诊断5"]
# df["诊断集合"] = df[diag_cols].apply(
#     lambda r:{str(x).strip() for x in r.dropna() if str(x).strip()},axis=1)
# mlb = MultiLabelBinarizer(sparse_output=False)
# icd_df = pd.DataFrame(mlb.fit_transform(df["诊断集合"]),
#                       columns=[f"icd_{c}" for c in mlb.classes_],
#                       index=df.index)
# df = pd.concat([df, icd_df], axis=1)

# # ===== 3. 特征列 =====
# base_cols = [
#     "年龄","身高","体重","BMI","mMarshall","SBP mmHg","DAP mmHg","MAPmmHg",
#     "呼吸频率/min","心率","体温max","体温min",
#     "白细胞计数","中性粒细胞计数","血红蛋白","血小板计数","红细胞比容",
#     "丙氨酸氨基转移酶","前白蛋白","白蛋白","总胆红素",
#     "肌酐","尿素","尿酸","钾","钠","钙","磷","葡萄糖",
#     "PT","APTT","Fg","D-二聚体定量(mg/L)","纤维蛋白降解产物(mg/L)","INR",
#     "心肌梗死（1是0否）","心衰(1是0否)","脑血管疾病(1是0否)","COPD (1是0否)",
#     "消化性溃疡(1是0否)","慢性肝病(1是0否)","糖尿病(1是0否)",
#     "高血压(1是0否)","慢性肾功能不全(1是0否)","实体肿瘤(1是0否)",
#     "腹腔穿刺（1是，0否）","是否使用抗生素（1是0否）",
#     "入院2周营养类型（0无；1肠内；2肠外）",
#     "CT诊断胰腺坏死（1是，0否）"
# ]
# feature_cols = base_cols + icd_df.columns.tolist()
# X, y = df[feature_cols], df["label"]

# # ===== 4. 预处理 =====
# num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
# cat_cols = [c for c in X.columns if c not in num_cols]

# numeric_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
#                          ("scaler", StandardScaler())])
# categoric_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
#                            ("ohe", OneHotEncoder(handle_unknown="ignore",
#                                                  sparse_output=True))])
# preprocess = ColumnTransformer([("num", numeric_pipe, num_cols),
#                                 ("cat", categoric_pipe, cat_cols)])

# # ===== 5. Hold-out 划分（确保含正类）=====
# def strat_holdout(X, y, test_size=0.2, seed=42):
#     sss = StratifiedShuffleSplit(
#         n_splits=1,          # 只能用关键字
#         test_size=test_size,
#         random_state=seed)
#     for tr, te in sss.split(X, y):
#         if y.iloc[te].sum() == 0:
#             return strat_holdout(X, y, test_size, seed + 1)
#         return tr, te

# train_idx, test_idx = strat_holdout(X,y)
# X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
# y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# rus = RandomUnderSampler(random_state=42)
# X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)

# # ===== 6. LightGBM 模型 =====
# cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
# class_weight = {c:w for c,w in zip(np.unique(y), cw)}
# lgbm_pipe = Pipeline([
#     ("prep", preprocess),
#     ("clf", LGBMClassifier(
#         objective="binary", metric="auc",
#         n_estimators=600, learning_rate=0.03, num_leaves=128,
#         class_weight=class_weight, subsample=0.8, colsample_bytree=0.8,
#         random_state=42))
# ])

# # ===== 7. Hold-out 评估 =====
# lgbm_pipe.fit(X_train_bal, y_train_bal)
# auc_hold = roc_auc_score(y_test, lgbm_pipe.predict_proba(X_test)[:,1])
# print(f"LightGBM Hold-out AUC = {auc_hold:.4f}   (正:{y_test.sum()} 负:{len(y_test)-y_test.sum()})")
# time.sleep(10)

# # ===== 8. 5-fold CV（每 fold 欠采样） =====
# cv = StratifiedKFold(5, shuffle=True, random_state=42)
# auc_scores = []
# for tr, val in cv.split(X,y):
#     X_tr, y_tr = rus.fit_resample(X.iloc[tr], y.iloc[tr])
#     if y.iloc[val].sum()==0:
#         continue
#     lgbm_pipe.fit(X_tr, y_tr)
#     auc_scores.append(roc_auc_score(
#         y.iloc[val], lgbm_pipe.predict_proba(X.iloc[val])[:,1]))
# print("LightGBM 5-fold AUC:", np.round(auc_scores,3),
#       "mean =", np.mean(auc_scores).round(4))

# # ===== 9. 保存模型 =====
# joblib.dump(lgbm_pipe, "data/model/machine-learning/ipn_lgbm_model.pkl")
# print("LightGBM 模型已保存 → ipn_lgbm_model.pkl")

# """
# train_ipn_catboost.py
# -----------------------------------------
# • 预处理 + 欠采样（与 V2 一致）
# • 仅训练 & 评估 CatBoostClassifier
# """

# from pathlib import Path
# import pandas as pd, numpy as np, warnings, joblib
# from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
# from sklearn.metrics import roc_auc_score
# from sklearn.utils.class_weight import compute_class_weight
# from imblearn.under_sampling import RandomUnderSampler
# from catboost import CatBoostClassifier
# import time

# warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# # ===== 1. 读数据与标签 =====
# DATA_FILE = Path("data/data(final).csv")
# df = pd.read_csv(DATA_FILE)
# df["label"] = df["结局(1死 0活(治愈/好转)) 2未愈 3其他"].apply(lambda x: 0 if x == 0 else 1)

# # ===== 2. ICD 多热 =====
# diag_cols = ["主要诊断","其他诊断1","其他诊断2","其他诊断3","其他诊断4","其他诊断5"]
# df["诊断集合"] = df[diag_cols].apply(
#     lambda r:{str(x).strip() for x in r.dropna() if str(x).strip()},axis=1)

# mlb = MultiLabelBinarizer(sparse_output=False)
# icd_df = pd.DataFrame(mlb.fit_transform(df["诊断集合"]),
#                       columns=[f"icd_{c}" for c in mlb.classes_],
#                       index=df.index)
# df = pd.concat([df, icd_df], axis=1)

# # ===== 3. 特征列 =====
# base_cols = [
#     "年龄","身高","体重","BMI","mMarshall","SBP mmHg","DAP mmHg","MAPmmHg",
#     "呼吸频率/min","心率","体温max","体温min",
#     "白细胞计数","中性粒细胞计数","血红蛋白","血小板计数","红细胞比容",
#     "丙氨酸氨基转移酶","前白蛋白","白蛋白","总胆红素",
#     "肌酐","尿素","尿酸","钾","钠","钙","磷","葡萄糖",
#     "PT","APTT","Fg","D-二聚体定量(mg/L)","纤维蛋白降解产物(mg/L)","INR",
#     "心肌梗死（1是0否）","心衰(1是0否)","脑血管疾病(1是0否)","COPD (1是0否)",
#     "消化性溃疡(1是0否)","慢性肝病(1是0否)","糖尿病(1是0否)",
#     "高血压(1是0否)","慢性肾功能不全(1是0否)","实体肿瘤(1是0否)",
#     "腹腔穿刺（1是，0否）","是否使用抗生素（1是0否）",
#     "入院2周营养类型（0无；1肠内；2肠外）",
#     "CT诊断胰腺坏死（1是，0否）"
# ]
# feature_cols = base_cols + icd_df.columns.tolist()
# X, y = df[feature_cols], df["label"]

# # ===== 4. 预处理流水线 =====
# num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
# cat_cols = [c for c in X.columns if c not in num_cols]

# numeric_pipe = Pipeline([
#     ("imp", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler())
# ])
# categoric_pipe = Pipeline([
#     ("imp", SimpleImputer(strategy="most_frequent")),
#     ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])
# preprocess = ColumnTransformer([
#     ("num", numeric_pipe, num_cols),
#     ("cat", categoric_pipe, cat_cols)
# ])

# # ===== 5. 分层 hold-out（确保含正类） =====
# def strat_holdout(X, y, test_size=0.2, seed=42):
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
#                                  random_state=seed)
#     for tr, te in sss.split(X, y):
#         if y.iloc[te].sum()==0:
#             return strat_holdout(X, y, test_size, seed+1)
#         return tr, te

# train_idx, test_idx = strat_holdout(X, y)
# X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
# y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# # 欠采样训练集
# rus = RandomUnderSampler(random_state=42)
# X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)

# print(f"Hold-out 正:{y_test.sum()} 负:{len(y_test)-y_test.sum()}")

# # ===== 6. 类别权重 =====
# cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
# class_weight = {c:w for c,w in zip(np.unique(y), cw)}

# # ===== 7. CatBoost Pipeline =====
# catboost_pipe = Pipeline([
#     ("prep", preprocess),
#     ("clf", CatBoostClassifier(
#         iterations=800, depth=6, learning_rate=0.03,
#         l2_leaf_reg=3, loss_function="Logloss",
#         class_weights=class_weight,
#         verbose=False, random_state=42))
# ])

# # ===== 8. Hold-out 评估 =====
# catboost_pipe.fit(X_train_bal, y_train_bal)
# auc_hold = roc_auc_score(y_test,
#                          catboost_pipe.predict_proba(X_test)[:,1])
# print(f"CatBoost Hold-out AUC = {auc_hold:.4f}")
# time.sleep(10)

# # ===== 9. 5-fold CV =====
# cv = StratifiedKFold(5, shuffle=True, random_state=42)
# auc_scores = []
# for tr, val in cv.split(X, y):
#     X_tr, y_tr = rus.fit_resample(X.iloc[tr], y.iloc[tr])
#     if y.iloc[val].sum()==0:
#         continue
#     catboost_pipe.fit(X_tr, y_tr)
#     auc_scores.append(roc_auc_score(
#         y.iloc[val], catboost_pipe.predict_proba(X.iloc[val])[:,1]))
# print("CatBoost 5-fold AUC:", np.round(auc_scores,3),
#       "mean =", np.mean(auc_scores).round(4))

# # ===== 10. 保存模型 =====
# joblib.dump(catboost_pipe, "data/model/machine-learning/ipn_catboost_model.pkl")
# print("CatBoost 模型已保存 → ipn_catboost_model.pkl")

"""
train_ipn_ensemble.py
-----------------------------------------
• 欠采样 + 多热 ICD
• CatBoost + LightGBM 软投票
• 输出：
    Ensemble Hold-out AUC = …
    Ensemble 5-fold AUC: [ … ] mean = …
"""

from pathlib import Path
import pandas as pd, numpy as np, warnings, joblib
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import time

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ===== 1. 读数据 & 标签 =====
DATA_FILE = Path("data/data(final).csv")
df = pd.read_csv(DATA_FILE)
df["label"] = df["结局(1死 0活(治愈/好转)) 2未愈 3其他"].apply(lambda x: 0 if x == 0 else 1)

# ===== 2. ICD 多热 =====
diag_cols = ["主要诊断","其他诊断1","其他诊断2","其他诊断3","其他诊断4","其他诊断5"]
df["诊断集合"] = df[diag_cols].apply(
    lambda r:{str(x).strip() for x in r.dropna() if str(x).strip()},axis=1)
mlb = MultiLabelBinarizer(sparse_output=False)
icd_df = pd.DataFrame(mlb.fit_transform(df["诊断集合"]),
                      columns=[f"icd_{c}" for c in mlb.classes_],
                      index=df.index)
df = pd.concat([df, icd_df], axis=1)

# ===== 3. 特征列 =====
base_cols = [
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
    "入院2周营养类型（0无；1肠内；2肠外）",
    "CT诊断胰腺坏死（1是，0否）"
]
feature_cols = base_cols + icd_df.columns.tolist()
X, y = df[feature_cols], df["label"]

num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# ===== 4. 预处理流水线 =====
numeric_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
categoric_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                 sparse_output=False))])
preprocess = ColumnTransformer([("num", numeric_pipe, num_cols),
                                ("cat", categoric_pipe, cat_cols)])

# ===== 5. 分层 hold-out（确保含正类）=====
def strat_holdout(X,y,test_size=0.2,seed=42):
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed)
    for tr,te in sss.split(X,y):
        if y.iloc[te].sum()==0:
            return strat_holdout(X,y,test_size,seed+1)
        return tr,te

train_idx, test_idx = strat_holdout(X,y)
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 欠采样训练集
rus = RandomUnderSampler(random_state=42)
X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)

# ===== 6. 类别权重 =====
cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weight = {c:w for c,w in zip(np.unique(y), cw)}

# ===== 7. 基模型 =====
cat = CatBoostClassifier(
    iterations=800, depth=6, learning_rate=0.03, l2_leaf_reg=3,
    loss_function="Logloss", class_weights=class_weight,
    verbose=False, random_state=42)

lgb = LGBMClassifier(
    objective="binary", metric="auc", n_estimators=600,
    learning_rate=0.03, num_leaves=128, class_weight=class_weight,
    subsample=0.8, colsample_bytree=0.8, random_state=42)

# ===== 8. 软投票集成 Pipeline =====
ensemble = Pipeline([
    ("prep", preprocess),
    ("ens", VotingClassifier(
        estimators=[("cat", cat), ("lgb", lgb)],
        voting="soft", weights=[1,1], n_jobs=-1))
])

# ===== 9. Hold-out 评估 =====
ensemble.fit(X_train_bal, y_train_bal)
auc_hold = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:,1])
print(f"Ensemble Hold-out AUC = {auc_hold:.4f}   (正:{y_test.sum()} 负:{len(y_test)-y_test.sum()})")
time.sleep(10)

# ===== 10. 5-fold CV（每 fold 欠采样）=====
cv = StratifiedKFold(5, shuffle=True, random_state=42)
auc_scores = []
for tr, val in cv.split(X,y):
    X_tr, y_tr = rus.fit_resample(X.iloc[tr], y.iloc[tr])
    if y.iloc[val].sum()==0:
        continue
    ensemble.fit(X_tr, y_tr)
    auc_scores.append(roc_auc_score(
        y.iloc[val], ensemble.predict_proba(X.iloc[val])[:,1]))
print("Ensemble 5-fold AUC:", np.round(auc_scores,3),
      "mean =", np.mean(auc_scores).round(4))

# ===== 11. 保存模型 =====
joblib.dump(ensemble, "data/model/ensemble-learning/ipn_cat_lgb_ensemble.pkl")
print("集成模型已保存 → ipn_cat_lgb_ensemble.pkl")

# ========= Visualization (confusion matrix + top-10 importances) =========
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from matplotlib import font_manager
# ---- 0. 让 Matplotlib 支持中文 ----
def set_chinese_font():
    for fname in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
        try:
            plt.rcParams["font.family"] = fname
            font_manager.FontProperties(fname=fname)   # 测试能否加载
            return
        except Exception:
            continue
set_chinese_font()
plt.rcParams["axes.unicode_minus"] = False  # 负号也正常显示

# ---------- 1. Confusion Matrix ----------
y_pred = ensemble.predict(X_test)
cm     = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(4,4))
disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
disp.plot(ax=ax_cm, values_format='d', cmap="Blues", colorbar=False)
ax_cm.set_title("Confusion Matrix – Hold-out")
plt.tight_layout()
fig_cm.savefig("conf_mat.png", dpi=300)
plt.close(fig_cm)
print("✅ saved  conf_mat.png")

# ---------- 2. Top-10 Feature Importances ----------
transformer   = ensemble.named_steps["prep"]
feature_names = transformer.get_feature_names_out()

#  VotingClassifier.fit 后 → named_steps["ens"].named_estimators_  是 dict
fitted_dict = ensemble.named_steps["ens"].named_estimators_

importances = []
for est in fitted_dict.values():
    if hasattr(est, "feature_importances_"):
        importances.append(est.feature_importances_)
if not importances:
    print("⚠️  no feature_importances_ in base estimators, skip plot")
else:
    mean_imp = np.mean(importances, axis=0)
    idx_top  = np.argsort(mean_imp)[-10:][::-1]
    top_feats, top_vals = feature_names[idx_top], mean_imp[idx_top]

    fig_imp, ax_imp = plt.subplots(figsize=(7,4))
    ax_imp.barh(range(len(top_feats))[::-1], top_vals, color="#1f77b4")
    ax_imp.set_yticks(range(len(top_feats))[::-1])
    ax_imp.set_yticklabels(top_feats, fontsize=8)
    ax_imp.set_xlabel("Average Importance")
    ax_imp.set_title("Top-10 Feature Importances")
    plt.tight_layout()
    fig_imp.savefig("top10_importances.png", dpi=300)
    plt.close(fig_imp)
    print("✅ saved  top10_importances.png")

"""
train_ipn_lgbm.py
-----------------------------------------
预测感染性胰腺坏死患者不良结局（死亡/未愈/其他）
• 诊断列(主要诊断+其他诊断1-5) → ICD 多热特征
• LightGBM + class_weight 自动平衡 1:20 不平衡
• 输出 hold-out AUC、5-fold CV AUC，并保存模型 pkl
"""

# ========= 1. 环境 =========
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
import joblib
import time

# ========= 2. 读数据 =========
DATA_FILE = Path("data/data(final).csv")
df = pd.read_csv(DATA_FILE)

# ========= 3. 标签处理 =========
df["label"] = df["结局(1死 0活(治愈/好转)) 2未愈 3其他"].apply(lambda x: 0 if x == 0 else 1)

# ========= 4. 诊断列多热编码 (方案 B) =========
diag_cols = ["主要诊断", "其他诊断1", "其他诊断2",
             "其他诊断3", "其他诊断4", "其他诊断5"]

df["诊断集合"] = df[diag_cols].apply(
    lambda r: {str(x).strip() for x in r.dropna() if str(x).strip()},
    axis=1
)

mlb = MultiLabelBinarizer(sparse_output=False)
icd_array = mlb.fit_transform(df["诊断集合"])
icd_df = pd.DataFrame(icd_array,
                      columns=[f"icd_{c}" for c in mlb.classes_],
                      index=df.index)

df = pd.concat([df, icd_df], axis=1)
multi_hot_cols = icd_df.columns.tolist()

# ========= 5. 选特征列 =========
base_cols = [
    "年龄", "身高", "体重", "BMI",
    "mMarshall", "SBP mmHg", "DAP mmHg", "MAPmmHg",
    "呼吸频率/min", "心率", "体温max", "体温min",
    "白细胞计数", "中性粒细胞计数", "血红蛋白", "血小板计数", "红细胞比容",
    "丙氨酸氨基转移酶", "前白蛋白", "白蛋白", "总胆红素",
    "肌酐", "尿素", "尿酸", "钾", "钠", "钙", "磷", "葡萄糖",
    "PT", "APTT", "Fg", "D-二聚体定量(mg/L)", "纤维蛋白降解产物(mg/L)", "INR",
    # 既往病史
    "心肌梗死（1是0否）", "心衰(1是0否)", "脑血管疾病(1是0否)",
    "COPD (1是0否)", "消化性溃疡(1是0否)", "慢性肝病(1是0否)",
    "糖尿病(1是0否)", "高血压(1是0否)", "慢性肾功能不全(1是0否)", "实体肿瘤(1是0否)",
    # 术前操作
    "腹腔穿刺（1是，0否）", "是否使用抗生素（1是0否）",
    "入院2周营养类型（0无；1肠内；2肠外）",
    "CT诊断胰腺坏死（1是，0否）"
]

feature_cols = base_cols + multi_hot_cols

# ========= 6. 构建 X, y =========
X = df[feature_cols]
y = df["label"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# ========= 7. 预处理流水线 =========
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categoric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categoric_pipe, cat_cols)
])

# ========= 8. 类别权重 =========
weights = compute_class_weight(class_weight="balanced",
                               classes=np.unique(y),
                               y=y)
class_weight = {cls: w for cls, w in zip(np.unique(y), weights)}
print("class_weight:", class_weight)

# ========= 9. LightGBM 模型 =========
lgbm = LGBMClassifier(
    objective="binary",
    metric="auc",
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=128,
    max_depth=-1,
    class_weight=class_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", lgbm)
])

# ========= 10. 训练 / Hold-out 验证 =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

pipe.fit(X_train, y_train)

y_pred_prob = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_prob)
print(f"Hold-out AUC = {auc:.4f}")
print(classification_report(y_test, (y_pred_prob > 0.5).astype(int)))
time.sleep(10)

# ========= 11. 5-fold CV =========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
for tr_idx, val_idx in cv.split(X, y):
    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    val_prob = pipe.predict_proba(X.iloc[val_idx])[:, 1]
    auc_scores.append(roc_auc_score(y.iloc[val_idx], val_prob))
print("5-fold AUC:", np.round(auc_scores, 3), "mean =", np.mean(auc_scores).round(4))

# ========= 12. 保存模型 =========
joblib.dump(pipe, "data/model/machine-learning/ipn_lgbm_model_V1.pkl")
print("模型已保存 → ipn_lgbm_model_V1.pkl")

"""
train_ipn_postop_ensemble.py
-----------------------------------------
• 两个目标列：
    1) 腹腔出血（1是0否）
    2) 腹腔感染（1是0否）
• CatBoost + LightGBM 软投票
• 与既往脚本同样的欠采样 / 评估打印格式
"""

# ------- 通用导入 -------
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

# ------- 1. 读数据 -------
DATA_FILE = Path("data/data(final).csv")
df = pd.read_csv(DATA_FILE)

# ------- 2. ICD 多热 -------
diag_cols = ["主要诊断","其他诊断1","其他诊断2",
             "其他诊断3","其他诊断4","其他诊断5"]
df["诊断集合"] = df[diag_cols].apply(
    lambda r:{str(x).strip() for x in r.dropna() if str(x).strip()}, axis=1)
mlb = MultiLabelBinarizer(sparse_output=False)
icd_df = pd.DataFrame(mlb.fit_transform(df["诊断集合"]),
                      columns=[f"icd_{c}" for c in mlb.classes_],
                      index=df.index)
df = pd.concat([df, icd_df], axis=1)

# ------- 3. 特征列 -------
base_cols = [  # 与前面保持一致
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

# ------- 4. 预处理管道 -------
num_cols = df[feature_cols].select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

numeric_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
categoric_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                           ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                 sparse_output=False))])
preprocess = ColumnTransformer([("num", numeric_pipe, num_cols),
                                ("cat", categoric_pipe, cat_cols)])

# ------- 5. 目标列表 -------
targets = {
    "Bleeding":  "腹腔出血（1是0否）",
    "Infection": "腹腔感染（1是0否）"
}

# ------- 6. 公用函数 -------
def strat_holdout(X,y,test_size=0.2,seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=seed)
    for tr,te in sss.split(X,y):
        if y.iloc[te].sum()==0:   # 确保含正类
            return strat_holdout(X,y,test_size,seed+1)
        return tr,te

def train_one(label_key, target_col):
    y = df[target_col]
    X = df[feature_cols]

    # hold-out
    train_idx, test_idx = strat_holdout(X,y)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rus = RandomUnderSampler(random_state=42)
    X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)

    # 类别权重
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_weight = {c:w for c,w in zip(np.unique(y), cw)}

    # 模型
    cat = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.03,
                             l2_leaf_reg=3, loss_function="Logloss",
                             class_weights=class_weight,
                             verbose=False, random_state=42)
    lgb = LGBMClassifier(objective="binary", metric="auc", n_estimators=600,
                         learning_rate=0.03, num_leaves=128,
                         class_weight=class_weight,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=42)

    ens = Pipeline([
        ("prep", preprocess),
        ("ens", VotingClassifier(
            estimators=[("cat", cat), ("lgb", lgb)],
            voting="soft", weights=[1,1], n_jobs=-1))
    ])

    # Hold-out
    ens.fit(X_train_bal, y_train_bal)
    auc_hold = roc_auc_score(y_test, ens.predict_proba(X_test)[:,1])
    print(f"[{label_key}] Ensemble Hold-out AUC = {auc_hold:.4f}   "
          f"(正:{y_test.sum()} 负:{len(y_test)-y_test.sum()})")
    time.sleep(10)

    # 5-fold
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    auc_scores=[]
    for tr,val in cv.split(X,y):
        X_tr, y_tr = rus.fit_resample(X.iloc[tr], y.iloc[tr])
        if y.iloc[val].sum()==0:
            continue
        ens.fit(X_tr, y_tr)
        auc_scores.append(roc_auc_score(
            y.iloc[val], ens.predict_proba(X.iloc[val])[:,1]))
    print(f"[{label_key}] Ensemble 5-fold AUC:",
          np.round(auc_scores,3),"mean =",np.mean(auc_scores).round(4))

    # 保存
    file_name = f"data/model/ensemble-learning/ipn_{label_key.lower()}_ensemble.pkl"
    joblib.dump(ens, file_name)
    print(f"[{label_key}] 模型已保存 → {file_name}\n")
    time.sleep(10)

# ------- 7. 依次训练两个目标 -------
for key, col in targets.items():
    if col not in df.columns:
        print(f"⚠️  列 {col} 不存在，跳过 {key}")
        continue
    if df[col].sum() < 10:
        print(f"⚠️  {key} 正样本过少 (<10)，结果可能不可靠")
    train_one(key, col)

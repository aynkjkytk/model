# test_3models.py
# ---------------------------------------
# 同时预测 Bleeding / Infection / Outcome 3 个概率
# ICD 六列一次性输入：空格分隔
# 其他特征逐列输入（回车跳过 = NaN）

import sys, joblib, pandas as pd, numpy as np
from pathlib import Path

# ---------- 模型路径 ----------
MODEL_DIR = Path("data/model")
MODELS = {
    "Bleeding":  MODEL_DIR / "ensemble-learning/ipn_bleeding_ensemble.pkl",
    "Infection": MODEL_DIR / "ensemble-learning/ipn_infection_ensemble.pkl",
    "Outcome":   MODEL_DIR / "ensemble-learning/ipn_blend_bundle.pkl"   # ← 融合版
}

# ---------- 加载模型 ----------
loaded = {}
for name, path in MODELS.items():
    if not path.exists():
        sys.exit(f"❌ 找不到模型文件：{path}")
    loaded[name] = joblib.load(path)

# ---------- 收集所有特征列 ----------
def feat_names(model):
    """从 Pipeline 或 bundle 中拿特征列名"""
    if isinstance(model, dict):            # Outcome bundle
        return model["weighted_lgb"].named_steps["prep"].feature_names_in_
    else:                                  # Pipeline
        return model.named_steps["prep"].feature_names_in_

all_cols  = sorted({c for m in loaded.values() for c in feat_names(m)})
icd_cols  = [c for c in all_cols if c.startswith("icd_")]
other_cols = [c for c in all_cols if not c.startswith("icd_")]

# ---------- 交互输入 ----------
print("\n🔹 请输入 6 个 ICD 编码（主要诊断 + 其他诊断 1-5），用空格分隔；不足可留空：")
codes = input("ICD 编码: ").strip().split()
codes = [c.strip() for c in codes if c.strip()]

print("\n🔹 其余临床特征：直接回车 = 缺失")
row = {}
for col in other_cols:
    val = input(f"{col}: ").strip()
    row[col] = np.nan if val == "" else float(val)

# 多热 ICD → one-hot
for ic in icd_cols:
    row[ic] = 1 if ic.split("icd_")[1] in codes else 0

df = pd.DataFrame([row], columns=all_cols)

# ---------- 逐模型预测 ----------
print("\n===========  预测结果  ===========")
for name, model in loaded.items():

    if name != "Outcome":                               # 普通 pipeline
        prob = model.predict_proba(df[feat_names(model)])[:, 1][0]

    else:                                               # 融合 bundle
        alpha   = model["alpha"]
        thresh  = model.get("threshold", 0.5)           # 若无则默认 0.5
        lgbm    = model["weighted_lgb"]
        voting  = model["undersample_voting"]
        feats   = feat_names(model)

        prob_l = lgbm.predict_proba(df[feats])[:, 1][0]
        prob_v = voting.predict_proba(df[feats])[:, 1][0]
        prob   = alpha * prob_l + (1 - alpha) * prob_v
        risk_tag = "⚠️ 高风险" if prob >= thresh else "✔️ 低风险"

    # ---------- 打印 ----------
    if name == "Outcome":
        print(f"{name:<9}:  概率 = {prob:.4f}   →  {risk_tag}")
    else:
        print(f"{name:<9}:  概率 = {prob:.4f}")
print("=================================")

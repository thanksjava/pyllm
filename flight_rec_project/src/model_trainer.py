import pandas as pd
import numpy as np  # <--- 修复点：必须导入 numpy
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class RecommendationModel:
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )

    def train(self, df: pd.DataFrame):
        if df.empty:
            print("Error: Training dataframe is empty.")
            return 0

        X = df[self.feature_cols]
        y = df['label']

        # 检查是否同时存在 0 和 1 标签，增加健壮性
        if len(np.unique(y)) < 2:
            print("Warning: Only one class present in label. AUC score is not defined for single class.")
            return 0.5

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"开始训练模型，训练样本数: {len(X_train)}...")
        self.model.fit(X_train, y_train)

        # 计算预测概率
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        return auc

    def save(self, model_path='models/flight_rec_model.joblib'):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"✅ 模型已成功保存至: {model_path}")
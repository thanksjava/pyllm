import pandas as pd  # <--- 修复点：必须先导入 pandas 才能使用 pd.DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os


class RecommendationModel:
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            class_weight='balanced',
            random_state=42
        )

    def train(self, df: pd.DataFrame):
        # 确保标签列存在
        if 'label' not in df.columns:
            raise ValueError("Dataframe must contain a 'label' column for training.")

        X = df[self.feature_cols]
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"开始训练模型... 训练集样本量: {len(X_train)}")
        self.model.fit(X_train, y_train)

        # 离线评估
        probs = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        return auc

    def save(self, path='models/flight_model_v1.joblib'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"✅ 模型成功保存至: {path}")
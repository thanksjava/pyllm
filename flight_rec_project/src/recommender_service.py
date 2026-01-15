import pandas as pd
import joblib
from .feature_engineering import FeatureProcessor


class RecommenderService:
    def __init__(self, model_path=None):
        # 实验组：加载已训练的模型
        if model_path:
            self.model = joblib.load(model_path)
            self.processor = FeatureProcessor()
        else:
            self.model = None

    def get_control_recommendations(self, candidates: pd.DataFrame, top_n=3):
        """对照组逻辑：简单规则（同航线最低价）"""
        if candidates.empty:
            return candidates
        # 按价格升序排列
        result = candidates.sort_values(by='total_price', ascending=True).head(top_n)
        result['rec_type'] = 'control_lowest_price'
        return result

    def get_test_recommendations(self, candidates: pd.DataFrame, feature_cols, top_n=3):
        """实验组逻辑：模型打分排序"""
        if self.model is None or candidates.empty:
            return self.get_control_recommendations(candidates, top_n)

        # 1. 实时特征加工
        processed_df = self.processor.process(candidates.copy())

        # 2. 调用模型预测点击概率
        scores = self.model.predict_proba(processed_df[feature_cols])[:, 1]
        candidates['predict_score'] = scores

        # 3. 按得分降序排列
        result = candidates.sort_values(by='predict_score', ascending=False).head(top_n)
        result['rec_type'] = 'test_ml_model'
        return result
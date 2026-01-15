import pandas as pd
import os
from src.feature_engineering import FeatureProcessor
from src.model_trainer import RecommendationModel


def main():
    # 1. 路径配置
    DATA_PATH = 'data/processed/1_.csv'

    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到数据文件: {DATA_PATH}")
        return

    # 2. 加载宽表
    print(f"Loading wide table from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # 3. 定义特征列表 (必须与 feature_engineering 生成的一致)
    features = [
        'price_ratio', 'price_diff', 'is_same_airline',
        'transfer_diff', 'total_price', 'adv_book_time',
        'go_transfer_count', 'resource_type_idx'
    ]

    # 4. 特征加工 - 重要：必须接收返回值
    processor = FeatureProcessor()
    processed_df = processor.process(df)

    # 5. 再次确认特征是否都存在，不存在则补齐 (防止 KeyError)
    for f in features:
        if f not in processed_df.columns:
            processed_df[f] = 0

    # 6. 模型训练
    trainer = RecommendationModel(feature_cols=features)
    auc_score = trainer.train(processed_df)

    print("-" * 30)
    print(f"✅ 训练成功! 离线评估 AUC: {auc_score:.4f}")
    print("-" * 30)

    trainer.save('models/flight_rec_model_v1.joblib')


if __name__ == "__main__":
    main()
import pandas as pd
from src.feature_engineering import FeatureProcessor
from src.model_trainer import RecommendationModel


def main():
    # 1. 配置路径
    PROCESSED_DATA_PATH = 'data/processed/1_.csv'

    # 2. 加载宽表
    print(f"Loading wide table from {PROCESSED_DATA_PATH}...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # 3. 特征定义
    # 包含您宽表中固有的特征和 FeatureProcessor 衍生的特征
    features = [
        'price_ratio', 'price_diff', 'is_same_airline',
        'transfer_diff', 'total_price', 'adv_book_time',
        'go_transfer_count', 'resource_type_idx'
    ]

    # 4. 执行特征工程
    processor = FeatureProcessor(features)
    processed_df = processor.process(df)

    # 5. 训练与模型持久化
    trainer = RecommendationModel(feature_cols=features)
    auc_score = trainer.train(processed_df)

    print("-" * 30)
    print(f"Model Training Results:")
    print(f"Offline AUC Score: {auc_score:.4f}")
    print("-" * 30)

    trainer.save()


if __name__ == "__main__":
    main()
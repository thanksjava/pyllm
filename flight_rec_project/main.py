import pandas as pd
from src.feature_engineering import FeatureProcessor
from src.model_trainer import RecommendationModel


def main():
    # 1. 加载数据 (假设之前SQL生成的宽表已存在)
    try:
        df = pd.read_csv('data/processed/1_.csv')
    except FileNotFoundError:
        print("未找到宽表数据，请确保已运行SQL清洗脚本并导出为csv。")
        return

    # 2. 特征加工
    processor = FeatureProcessor()
    processed_df = processor.process(df)

    # 3. 定义特征集
    features = [
        'price_ratio', 'price_diff', 'is_same_airline',
        'transfer_diff', 'go_fly_time', 'adv_book_time',
        'resource_type_idx', 'cabin_class_code_idx'
    ]

    # 4. 训练与保存
    trainer = RecommendationModel(feature_cols=features)
    auc_score = trainer.train(processed_df)

    print(f"模型训练完成! 离线 AUC: {auc_score:.4f}")

    trainer.save()


if __name__ == "__main__":
    main()
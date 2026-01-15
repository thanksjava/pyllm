import pandas as pd
import numpy as np


class FeatureProcessor:
    def __init__(self):
        self.cat_mappings = {}

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """从宽表生成训练所需的特征"""
        # 1. 差异化核心特征
        # 计算推荐价格与原失败价格的比例和差值
        df['price_ratio'] = df['total_price'] / (df['origin_price'] + 0.1)
        df['price_diff'] = df['total_price'] - df['origin_price']

        # 航司一致性特征
        df['is_same_airline'] = (df['go_main_cxr'] == df['origin_airline']).astype(int)

        # 航程效率特征：转机次数差
        df['transfer_diff'] = df['go_transfer_count'] - df['origin_transfer']

        # 2. 基础数值清洗
        df['adv_book_time'] = df['adv_book_time'].fillna(0)
        df['go_fly_time'] = df['go_fly_time'].fillna(df['go_fly_time'].median())

        # 3. 类别特征编码 (Simple Label Encoding)
        for col in ['resource_type', 'cabin_class_code']:
            df[f'{col}_idx'] = pd.factorize(df[col])[0]

        return df
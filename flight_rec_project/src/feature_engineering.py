import pandas as pd
import numpy as np


class FeatureProcessor:
    def __init__(self, features):
        self.features = features

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """对宽表进行最终特征加工"""
        if df.empty:
            return df

        # 1. 核心差异特征计算（基于宽表中的 origin_ 字段）
        # 价格比率：衡量备选航班比原航班贵/便宜的幅度
        if 'total_price' in df.columns and 'origin_price' in df.columns:
            df['price_ratio'] = df['total_price'] / (df['origin_price'] + 0.1)
            df['price_diff'] = df['total_price'] - df['origin_price']

        # 航司一致性
        if 'go_main_cxr' in df.columns and 'origin_airline' in df.columns:
            df['is_same_airline'] = (df['go_main_cxr'] == df['origin_airline']).astype(int)

        # 转机次数差异
        if 'go_transfer_count' in df.columns and 'origin_transfer' in df.columns:
            df['transfer_diff'] = df['go_transfer_count'] - df['origin_transfer']

        # 2. 类别特征编码（处理 resource_type 等字符串）
        for col in ['resource_type', 'cabin_class_code', 'go_main_cxr']:
            if col in df.columns:
                df[f'{col}_idx'] = pd.factorize(df[col].astype(str))[0]

        # 3. 填充缺失值并返回
        return df.fillna(0)
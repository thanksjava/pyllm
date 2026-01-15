import pandas as pd
import numpy as np


class FeatureProcessor:
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        # 创建副本，避免对原数据产生 SettingWithCopyWarning
        df = df.copy()

        if df.empty:
            return df

        # 1. 价格差异特征 (核心中的核心)
        # 增加容错：如果宽表中没有这些列，初始化为0
        if 'total_price' in df.columns and 'origin_price' in df.columns:
            df['price_ratio'] = df['total_price'] / (df['origin_price'] + 0.1)
            df['price_diff'] = df['total_price'] - df['origin_price']
        else:
            df['price_ratio'] = 1.0
            df['price_diff'] = 0.0

        # 2. 航司一致性
        if 'go_main_cxr' in df.columns and 'origin_airline' in df.columns:
            df['is_same_airline'] = (df['go_main_cxr'] == df['origin_airline']).astype(int)
        else:
            df['is_same_airline'] = 0

        # 3. 转机次数差异
        if 'go_transfer_count' in df.columns and 'origin_transfer' in df.columns:
            df['transfer_diff'] = df['go_transfer_count'] - df['origin_transfer']
        else:
            df['transfer_diff'] = 0

        # 4. 类别特征数字化 (resource_type, cabin_class_code)
        for col in ['resource_type', 'cabin_class_code']:
            if col in df.columns:
                df[f'{col}_idx'] = pd.factorize(df[col].astype(str))[0]
            else:
                df[f'{col}_idx'] = -1

        # 5. 确保返回处理后的对象
        return df.fillna(0)
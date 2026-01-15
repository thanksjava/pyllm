import sys
import os
import pandas as pd

# 修复导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.recommender_service import RecommenderService


def run_debug_demo():
    # 1. 初始化服务
    model_path = os.path.join(parent_dir, 'models/flight_rec_model_v1.joblib')
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}，请先运行 main.py 训练模型")
        return

    service = RecommenderService(model_path=model_path)

    # 2. 模拟下单失败的原始航班 (Base)
    origin_flight = {
        'total_price': 2000.0,
        'go_main_cxr': 'HU',
        'go_transfer_count': 0,
        'origin_fly_time': 180
    }

    # 3. 模拟实时搜索到的候选航班 (补全缺失字段)
    candidates_data = [
        {
            'trip_id': 'F001_LowPrice',
            'total_price': 1500.0,
            'go_main_cxr': 'CZ',
            'go_transfer_count': 1,
            'adv_book_time': 5,
            'resource_type': 'GDS',
            'cabin_class_code': 'Y',
            'go_fly_time': 240
        },
        {
            'trip_id': 'F002_SameAirline',
            'total_price': 2100.0,
            'go_main_cxr': 'HU',
            'go_transfer_count': 0,
            'adv_book_time': 5,
            'resource_type': 'IBE',
            'cabin_class_code': 'Y',
            'go_fly_time': 180
        }
    ]
    df_candidates = pd.DataFrame(candidates_data)

    # 注入基准特征 (用于 FeatureProcessor 计算差异特征)
    df_candidates['origin_price'] = origin_flight['total_price']
    df_candidates['origin_airline'] = origin_flight['go_main_cxr']
    df_candidates['origin_transfer'] = origin_flight['go_transfer_count']
    df_candidates['origin_fly_time'] = origin_flight['origin_fly_time']

    # 4. 对比 A/B Test 结果
    print("\n" + "=" * 20 + " 对照组 (Control: Lowest Price) " + "=" * 20)
    print(service.get_control_recommendations(df_candidates.copy(), top_n=2)[['trip_id', 'total_price', 'rec_type']])

    print("\n" + "=" * 20 + " 实验组 (Test: ML Model Score) " + "=" * 20)

    # ⚠️ 特征列表必须与训练时完全一致
    features = [
        'price_ratio', 'price_diff', 'is_same_airline',
        'transfer_diff', 'total_price', 'adv_book_time',
        'go_transfer_count', 'resource_type_idx'
    ]

    try:
        recommendations = service.get_test_recommendations(df_candidates.copy(), features, top_n=2)
        print(recommendations[['trip_id', 'total_price', 'predict_score', 'rec_type']])
    except Exception as e:
        print(f"❌ 实验组推理失败: {e}")


if __name__ == "__main__":
    run_debug_demo()
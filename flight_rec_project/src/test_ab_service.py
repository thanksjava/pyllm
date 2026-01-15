import pandas as pd
from src.recommender_service import RecommenderService


def run_debug_demo():
    # 1. 初始化推荐服务 (加载已训练的模型)
    # 确保 models/ 目录下已存在 flight_rec_model_v1.joblib
    try:
        service = RecommenderService(model_path='models/flight_rec_model_v1.joblib')
    except Exception as e:
        print(f"模型加载失败，请先运行训练脚本: {e}")
        return

    # 2. 模拟业务场景：用户原本想买 HU7181，但下单失败了
    origin_flight = {
        'total_price': 1500.0,
        'go_main_cxr': 'HU',
        'go_transfer_count': 0
    }

    # 3. 模拟实时搜索到的候选航班列表 (Candidate List)
    # 包含价格、航司、转机次数等关键维度
    candidates_data = [
        {'trip_id': 'flight_001_low_price', 'total_price': 1200.0, 'go_main_cxr': 'CZ', 'go_transfer_count': 1,
         'adv_book_time': 7, 'resource_type': 'GDS', 'cabin_class_code': 'Y', 'go_fly_time': 300},
        {'trip_id': 'flight_002_same_cxr', 'total_price': 1600.0, 'go_main_cxr': 'HU', 'go_transfer_count': 0,
         'adv_book_time': 7, 'resource_type': 'IBE', 'cabin_class_code': 'Y', 'go_fly_time': 180},
        {'trip_id': 'flight_003_fastest', 'total_price': 2500.0, 'go_main_cxr': 'EK', 'go_transfer_count': 0,
         'adv_book_time': 7, 'resource_type': 'GDS', 'cabin_class_code': 'Y', 'go_fly_time': 150},
        {'trip_id': 'flight_004_mid_price', 'total_price': 1550.0, 'go_main_cxr': 'HU', 'go_transfer_count': 1,
         'adv_book_time': 7, 'resource_type': 'IBE', 'cabin_class_code': 'Y', 'go_fly_time': 240}
    ]
    df_candidates = pd.DataFrame(candidates_data)

    # 注入原失败航班信息作为特征工程的对比基准
    df_candidates['origin_price'] = origin_flight['total_price']
    df_candidates['origin_airline'] = origin_flight['go_main_cxr']
    df_candidates['origin_transfer'] = origin_flight['go_transfer_count']
    df_candidates['origin_fly_time'] = 180  # 假设原航班时长

    # 4. 模拟对照组逻辑 (Control Group: 价格优先)
    print("\n" + "=" * 20 + " 对照组 (Control: Lowest Price) " + "=" * 20)
    control_res = service.get_control_recommendations(df_candidates.copy(), top_n=2)
    print(control_res[['trip_id', 'total_price', 'rec_type']])

    # 5. 模拟实验组逻辑 (Test Group: 模型打分)
    # 需要定义的特征列名
    feature_cols = [
        'price_ratio', 'price_diff', 'is_same_airline',
        'transfer_diff', 'total_price', 'adv_book_time',
        'go_transfer_count', 'resource_type_idx'
    ]

    print("\n" + "=" * 20 + " 实验组 (Test: ML Model Score) " + "=" * 20)
    test_res = service.get_test_recommendations(df_candidates.copy(), feature_cols, top_n=2)
    print(test_res[['trip_id', 'total_price', 'predict_score', 'rec_type']])


if __name__ == "__main__":
    run_debug_demo()
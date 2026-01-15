from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from src.recommender_service import RecommenderService

app = FastAPI(title="国际机票下单失败重推荐服务")

# 初始化服务（加载模型）
# 假设 features 列表与训练时一致
FEATURE_COLS = [
    'price_ratio', 'price_diff', 'is_same_airline',
    'transfer_diff', 'total_price', 'adv_book_time',
    'go_transfer_count', 'resource_type_idx'
]
service = RecommenderService(model_path='models/flight_rec_model_v1.joblib')


class FlightCandidate(BaseModel):
    trip_id: str
    total_price: float
    go_main_cxr: str
    go_transfer_count: int
    adv_book_time: int
    resource_type: str
    # ... 其他字段


@app.post("/recommend")
async def recommend(group: str, origin_flight: dict, candidates: List[dict]):
    """
    group: 'control' 或 'test'
    """
    df_candidates = pd.DataFrame(candidates)

    # 注入原失败航班信息以便计算差异特征
    df_candidates['origin_price'] = origin_flight['total_price']
    df_candidates['origin_airline'] = origin_flight['go_main_cxr']
    df_candidates['origin_transfer'] = origin_flight['go_transfer_count']

    if group == 'test':
        res = service.get_test_recommendations(df_candidates, FEATURE_COLS)
    else:
        res = service.get_control_recommendations(df_candidates)

    return res[['trip_id', 'total_price', 'rec_type']].to_dict(orient='records')
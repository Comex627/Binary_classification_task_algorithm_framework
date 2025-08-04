import numpy as np
import pandas as pd

def map_willingness_to_score(probabilities, base_score=600, pdo=20, target_odds=5, target_prob=0.5):
    """
    将贷款意愿概率映射为得分（意愿越高，得分越高）
    
    参数:
        probabilities: 模型输出的贷款意愿概率数组（0-1之间）
        base_score: 基准分数（对应目标概率的分数）
        pdo: 概率翻倍的分数（每增加pdo分，意愿概率翻倍）
        target_odds: 目标比值（意愿客户/非意愿客户）
        target_prob: 对应基准分数的目标概率
        
    返回:
        scores: 映射后的得分（整数）
    """
    # 处理极端值，避免log(0)或log(1)
    eps = 1e-10
    probs = np.clip(probabilities, eps, 1 - eps)
    
    # 计算对数几率 log(odds) = log(p/(1-p))，这里p是贷款意愿概率
    log_odds = np.log(probs / (1 - probs))
    
    # 计算映射参数
    factor = pdo / np.log(2)  # PDO参数
    offset = base_score - factor * np.log(target_odds)
    
    # 计算得分（贷款意愿越高，log_odds越大，得分越高）
    scores = offset + factor * log_odds
    
    # 设置合理的分数范围（300-850）
    scores = np.clip(scores, 300, 700)
    
    return np.round(scores).astype(int)










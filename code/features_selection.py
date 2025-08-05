#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import warnings

# 忽略警告
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)



# 提取时间特征
def get_time_fe(df):
    df['day'] = df['date'].apply(lambda x: int(x[8:10]))
    df['hour'] = df['date'].apply(lambda x: int(x[11:13]))
    return df

# 时间分箱(当前代码未实际使用，保留定义)
def getSeg(x):
    if 0 <= x <= 3:
        return 1
    elif 4 <= x <= 12:
        return 2
    elif 13 <= x <= 18:
        return 3
    elif 19 <= x <= 23:
        return 1

# 交叉特征
cross_feature = []
def get_cross_fe(df):
    first_feature = ['B2', 'B3']
    second_feature = ['C1', 'C2', 'C3', 'D1', 'A1', 'A2', 'A3']
    for feat_1 in first_feature:
        for feat_2 in second_feature:
            col_name = f"cross_{feat_1}_and_{feat_2}"
            cross_feature.append(col_name)
            df[col_name] = df[feat_1].astype(str) + '_' + df[feat_2].astype(str)
    return df



#减少内存消耗，对int64和float64类型做处理


def downcast_int(data,_list):
    int_downcast = data.select_dtypes(include = ['int64']).columns
    for col in _list:
        data[col] =pd.to_numeric(data[col], downcast='integer')
        
    return data

def downcast_float(data,_list):
    float_downcast = data.select_dtypes(include = ['float64']).columns
    for col in _list:
        data[col] =pd.to_numeric(data[col], downcast='float')
        
    return data





# 唯一值统计特征
def get_nunique_1_fe(df):
    adid_nuq = ['hour', 'E1', 'E14', 'B2', 'B3']
    for feat in adid_nuq:
        gp1 = df.groupby('A2')[feat].nunique().reset_index().rename(
            columns={feat: f"A2_{feat}_nuq_num"})
        gp2 = df.groupby(feat)['A2'].nunique().reset_index().rename(
            columns={'A2': f"{feat}_A2_nuq_num"})
        df = pd.merge(df, gp1, how='left', on='A2')
        df = pd.merge(df, gp2, how='left', on=feat)
    return df

def get_nunique_2_fe(df):
    adid_nuq = ['E1', 'E14']
    for feat in adid_nuq:
        gp1 = df.groupby('hour')[feat].nunique().reset_index().rename(
            columns={feat: f"hour_{feat}_nuq_num"})
        gp2 = df.groupby(feat)['hour'].nunique().reset_index().rename(
            columns={'hour': f"{feat}_hour_nuq_num"})
        df = pd.merge(df, gp1, how='left', on='hour')
        df = pd.merge(df, gp2, how='left', on=feat)
    return df

def get_nunique_4_fe(df):
    adid_nuq = ['B2', 'B3']
    for feat in adid_nuq:
        gp1 = df.groupby('A1')[feat].nunique().reset_index().rename(
            columns={feat: f"A1_{feat}_nuq_num"})
        gp2 = df.groupby(feat)['A1'].nunique().reset_index().rename(
            columns={'A1': f"{feat}_A1_nuq_num"})
        df = pd.merge(df, gp1, how='left', on='A1')
        df = pd.merge(df, gp2, how='left', on=feat)
    return df

# 计数特征
def feature_count(data, features):
    new_feature = f"count_{'_'.join(features)}"
    if new_feature in data.columns:
        del data[new_feature]
    temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
    return data.merge(temp, 'left', on=features)




def process_features(train, test, label):
    """
    特征工程处理函数
    
    参数:
        train: 训练集DataFrame，包含ID和特征列
        test: 测试集DataFrame，包含ID和特征列
        label: 标签DataFrame，包含ID和label列
    
    返回:
        train_x: 处理后的训练集特征
        train_y: 训练集标签
        test_x: 处理后的测试集特征
    """
    # 数据合并
    train = train.merge(label, on='ID', how='left')
    test['label'] = -1  # 标记测试集标签
    data = pd.concat([train, test], ignore_index=True)
    

    
    # 应用特征工程
    data = get_time_fe(data)
    data = get_cross_fe(data)
    data = get_nunique_1_fe(data)
    data = get_nunique_2_fe(data)
    #data = get_nunique_4_fe(data)  # 补充原代码中遗漏的调用
    
    # 标签编码
    cate_feature = ['A1','A2','A3','B1','B2','B3','C1','C2','C3','E2','E3','E5','E7','E9','E10','E13','E16','E17','E19','E21','E22']
    cate_features = cate_feature + cross_feature
    for item in cate_features:
        data[item] = LabelEncoder().fit_transform(data[item].astype(str))  # 确保字符串类型
    
    # 处理计数特征
    for i in cross_feature:
        if data[i].nunique() > 5:
            data = feature_count(data, [i])
    
    # 比例特征
    label_feature = ['A2', 'A3', 'hour']
    data_temp = data[label_feature].copy()
    data_temp['cnt'] = 1
    df_feature = pd.DataFrame()
    for col in label_feature:
        df_feature[f"ratio_click_of_{col}"] = (
            data_temp[col].map(data_temp[col].value_counts()) / len(data) * 100
        ).astype(int)
    data = pd.concat([data, df_feature], axis=1)
    
    # 拆分数据
    train_df = data[data['label'] != -1].copy()
    test_df = data[data['label'] == -1].copy()
    
    # 特征筛选
    del_feature = ['ID', 'day', 'date', 'label', 'D2'] + cross_feature
    features = [i for i in train_df.columns if i not in del_feature]
    
    return train_df[features], train_df['label'].values, test_df[features],features
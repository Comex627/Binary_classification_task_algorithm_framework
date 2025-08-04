import numpy as np
import pandas as pd
from typing import Union
from sklearn.metrics import accuracy_score, auc, recall_score, roc_curve, roc_auc_score, average_precision_score, \
    precision_score
import math
from sklearn.metrics import precision_recall_curve,average_precision_score,f1_score,confusion_matrix
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 数值类型转换:讲字符串转为数字
def convert_to_numeric(df):

       df_convert = df.copy()
       object_cols = df_convert.dropna(how = 'all',axis = 1 ).select_dtypes(include = ['object']).columns
       # 数值转换
       for col in object_cols:
           df_convert[col] = pd.to_numeric(df_convert[col],errors = 'ignore')
       return df_convert

def cal_psi(expect_df,actual_df,bins = 10):

       # 分箱
       bins = pd.qcut( x = expect_df, q = bins, duplicates = 'drop').unique().categories

       # 分箱计数
       expect_freq = pd.cut(expect_df , bins = bins, include_lowest = True).value_counts(normalize = True,sort = False)
       actual_freq = pd.cut(actual_df, bins = bins, include_lowest = True).value_counts(normalize = True,sort = False)

       # 计算 PSI
       psi = 0.0
       for bin in bins:
           expect_bin_freq = expect_freq.get(bin,0)
           actual_bin_freq = actual_freq.get(bin,0)

           # 计算 PSI 分项
           if expect_bin_freq > 0 and actual_bin_freq > 0:
              psi += (actual_bin_freq - expect_bin_freq) * np.log(actual_bin_freq/expect_bin_freq)
           elif expect_bin_freq == 0 and actual_bin_freq > 0:
               psi += (actual_bin_freq - 0.000001) * np.log(actual_bin_freq - 0.000001)
           elif expect_bin_freq > 0 and actual_bin_freq == 0:
               psi += (0.000001 - expect_bin_freq) * np.log(0.000001 / expect_bin_freq)
           elif expect_bin_freq == 0 and actual_bin_freq == 0:
               psi += 0
       return psi


# 计算 IV
def cal_iv(feature,target,bins = 10):

    df = pd.DataFrame({'feature':feature,'target':target})
    df['feature'] = df['feature'].fillna('-99')

    # 分箱，空值单独分箱
    df1 = df[df['feature'] != '-99']
    df2 = df[df['feature'] == '-99']

    df1['binned'] = pd.cut(x=df1['target'], bins = bins,duplicates = 'drop')
    df2['binned'] = '-99'
    df = pd.concat([df1,df2],axis=0)
    # 计算各分箱中的好坏客户数
    iv_table = df.groupby('binned').agg({'target':['count','sum']}).reset_index()
    iv_table.columns = ['binned','total','bad']
    iv_table['good'] = iv_table['total'] - iv_table['bad']

    # 处理样本数量为 0 的分箱
    iv_table = iv_table[iv_table['total'] > 0]

    # 计算各分箱的占比和好坏比
    total_bad = iv_table['bad'].sum()
    total_good = iv_table['good'].sum()

    iv_table['bad_cnt'] = iv_table['bad']/total_bad
    iv_table['good_cnt'] = iv_table['good']/total_good

    iv_table['woe'] = np.log((iv_table['bad_cnt'] + 1e-10) / (iv_table['good_cnt'] + 1e-10))
    iv_table['iv'] = (iv_table['bad_cnt'] - iv_table['good_cnt']) * iv_table['woe']
    iv = iv_table['iv'].sum()

    return iv


# 计算多重共线性
def cal_vif(x):

    vif = pd.DataFrame()
    vif['feature'] = x.columns
    vif['VIF'] = [variance_inflation_factor(x.values,i) for i in range(x.shape[1])]

    return vif


# 计算 KS 值
def cal_ks(y_true,y_prob):

    fpr, tpr, thresholds = roc_curve(y_true,y_prob)
    ks = np.max(np.abs(tpr - fpr))

    return ks


# 概率转分值
def prob_to_score(prob,base_score = 500,pdo = 20):

    prob = np.array(prob)
    score = base_score - (pdo / np.log(2)) * np.log(prob / (1 - prob))

    return score


# 分段评估统计
def eval_score(df,score_col,label,bin_width = 5):

    # 创建分段
    df['bin'] = pd.cut(df[score_col]
                       ,bins = [- np.inf] + list(np.arange(df[score_col].min()
                                                            ,df[score_col].max() + bin_width
                                                            ,bin_width
                                                           )
                                                 )
                       ,include_lowest = True
                      )

    # 计算每个分段的统计信息
    result = df.groupby('bin').agg({'bin':['count'],'label':['sum']}).reset_index()
    result.columns = ['bin','bin_total','bad']
    result['good'] = result['bin_total'] - result['bad']

    # 计算累计样本数 和 样本占比
    result['cum_total'] = result['bin_total'].cumsum()
    result['cum_bad'] = result['bad'].cumsum()
    result['cum_good'] = result['good'].cumsum()
    result['cum_total_rate'] = result['cum_total'] / result['bin_total'].sum()
    result['cum_bad_rate'] = result['cum_bad'] / result['cum_total'].sum()
    result['good'] = result['cum_good'] / result['cum_total'].sum()

    result['recall'] = result['cum_bad'] / result['bad'].sum()
    result['precision'] = result['cum_bad'] / (result['cum_bad'] + result['good'])

    result['f1'] = 2 * (result['precision'] * result['recall']) / (result['precision'] + result['recall'])

    # 计算 lift
    base_bad_rate = result['bad'].sum() / result['bin_total'].sum()
    result['lift'] = result['cum_bad_rate'] / base_bad_rate

    result = result.rename(columns = {'bin':'分数区间'
                                      ,'bin_total':'区间样本总量'
                                      ,'bad':'label_1样本量'
                                      ,'good':'label_0样本量'
                                      ,'cum_total':'累计样本量'
                                      ,'cum_bad':'累计label_1样本量'
                                      ,'cum_good':'累计label_0样本量'
                                      ,'cum_bad_rate':'label_1占比'
                                      ,'cum_good_rate':'label_0占比'
                                      ,'cum_total_rate':'累计样本占比'
                                      ,'recall':'召回率'
                                      ,'precision':'准确率'
                                      ,'f1':'F1-score'
                                      ,'lift':'Lift'
                                      }
                           )

    return result


# 异常值盖帽处理
def apply_cap(df,features,lower_quantile = 0.01,upper_quantile = 0.99):

    df_cap = df.copy()

    for feature in features:
        lower_bounded = round(df_cap[feature].quantile(lower_quantile),4)
        upper_bounded = round(df_cap[feature].quantile(upper_quantile),4)

        df_cap[feature] = df_cap[feature].clip(lower = lower_bounded, upper = upper_bounded)

    return df_cap



# -1111 中位数填充
# def median_fillna(df):
#
#     cols = df.columns[(df == -1111).any()]
#     for col in cols:
#         if df[col].dtype in ['int64','float64'] and df[df[col] == -1111].shape[0] > 1:
#             if df[df[col] >0 ].shape[0] > 0:
#                 median = df[df[col] > 0][col].median()
#             else:
#                 median = 0
#             df[col] = df[col].replace(-1111,median)
#             print(col + ':' + str(median))
#     return df



# 中位数填充 -1111
def median_fillna(df):

    cols = df.columns[(df == -1111).any()]

    for col in cols:
        if df[col].dtype in ['int64','float64']:
            median = df[col].replace(-1111 , np.nan).median()
            df[col] = df[col].replace(-1111,median)

    return df


# 中英文特征转换
def feature_to_chinese(feature_names,translation_csv = '.csv'):

    translations = pd.read_csv(translation_csv)

    # 创建英文到中文的字典
    english_to_chinese = dict(zip(translations[translations['English'],translations['Chinese']]))

    # 初始化结果列表
    results = []

    for feature in feature_names:
        # 获取中文名称，如果没有中文名称，使用'/'
        chinese_name  = english_to_chinese.get(feature,'/')
        result.append({'feature_name':feature,'中文名称':chinese_name})

    result_df = pd.DataFrame(results)

    return result_df




# TopN 统计
def cal_metrics_to_topn(y_prob,y_true,topn_list = [500,1000,3000,5000]):

    # 创建 DataFrame 来保护用户 ID、预测概率和真实标签
    df = pd.DataFrame({'y_true':y_true,'y_prob':y_prob})
    # 排序
    df = df.sort_values(by ='y_prob',ascending=False).reset_index(drop = True)

    results = []

    for topn in topn_list:
        # 获取当前 top_n 的概率阈值
        threadshold = df.iloc[topn - 1]['y_prob']

        # 获取 top_n 的子集
        df_topn = df.iloc[:topn]
        y_true_topn = df_topn['y_true']
        y_pred_topn = df_topn['y_prob'].apply(lambda x : 1 if x >= threadshold else 0)

        # 计算 TP、FP、TN、FN
        cm = confusion_matrix(np.array(y_true_topn),np.array(y_pred_topn)).ravel()

        if cm.shape == (4,):
            tn,fp,fn,tp = cm.ravel()
        elif cm.shape == (1,):
            if y_true_topn[0] == 0:
                tn = cm[0]
                fp = 0
                fn = 0
                tp = 0
            else:
                tp = cm[0]
                tn = 0
                fp = 0
                fn = 0
        # 计算精准率、召回率、F1 和 Lift
        T1 = pd.Series(y_true).sum()
        recall = tp / T1
        precision = precision_score(y_true_topn,y_pred_topn)
        f1 = 2 * precision * recall / (precision + recall)
        lift = (tp /topn) / (df['y_true'].sum() / len(df['y_true']))

        # 保存结果
        results.append({
            'TopN':topn,
            '边界阈值':threadshold,
            'TopN样本占比':'{:.2%}'.format(topn / pd.Series(y_true).shape[0]),
            '预测正确': tp,
            '预测错误': fp,
            '样本总量': df.shape[0],
            '总1数量': T1,
            '总1比例':'{:.2%}'.format(T1 / pd.Series(y_true).shape[0]),
            '精准率': '{:.2%}'.format(precision),
            '召回率': '{:.2%}'.format(recall),
            'F1': f1,
            'Lift': lift
        })

    # 转换结果为 DataFrame
    result_df = pd.DataFrame(results)
    return result_df


# 分概率统计
def cal_metrics_to_prob(y_prob,y_true,prob_list = [0.95,0.9,0.85,0.8,0.7,0.6]):

    # 创建 DataFrame 来保存用户 ID、预测概率和真实标签
    df = pd.DataFrame({'y_true':y_true,'y_prob':y_prob})

    # 排序
    df = df.sort_values('y_prob',ascending=False).reset_index(drop=True)

    results = []

    T1 = y_true.sum()
    for prob in prob_list:
        temp = df.query('y_prob >= @prob')
        if temp.shape[0] > 0:
            tp = temp.query('y_true == 1').shape[0]
            fp = temp.query('y_true == 0').shape[0]
            precision = tp / temp.shape[0]
            recall = tp / T1
            f1 = 2 * recall * precision / (recall + precision)
            lift = (tp / temp.shape[0]) / (df['y_true'].sum()) / len(df['y_true'])

        else:
            tp = 0
            fp = 0
            precision = 0
            recall = 0
            f1 = 0
            lift = 0

         # '{:.2%}'.format(temp.shape[0] / y_true.shape[0])

        # 保存结果
        results.append(
                       {
                         '概率阈值':prob,
                         '样本数量':temp.shape[0],
                         '样本占比':'{:.2%}'.format(temp.shape[0]),
                         '预测正确':tp,
                         '预测错误':fp,
                         '样本总量':df.shape[0],
                         '总1数量':T1,
                         '总1比例':'{:.2%}'.format(T1 / y_true.shape[0]),
                         '召回率':'{:.2%}'.format(recall),
                         '精准率':'{:.2%}'.format(precision),
                         'F1':f1,
                         'LIFT':round(lift,2)
                       }
                     )
    # 转换 结果 为 DataFrame
    result_df = pd.DataFrame(results)
    return result_df


def fun_gender(x):
    if x == '男性':
        return 1
    elif x == '女性':
        return 2
    else:
        return 0

def fun_ocp(x):
    if x == '不便分类的其他从业人员':
        return 1
    elif x == '农、林、牧、渔业生产及辅助人员':
        return 2
    elif x == '办事人员和有关人员':
        return 3
    elif x == '企业负责人':
        return 4
    elif x == '社会生产服务和生活服务人员':
        return 5
    elif x == '党的机关、国家机关、群众团体和社会组织、企事业单位负责人':
        return 6
    elif x == '专业技术人员':
        return 7
    elif x == '运输设备和通用工程机械操作人员及有关人员':
        return 8
    else:
        return 0

def fun_post(x):
    if x == '（副）厅/局级以上':
        return 1
    elif x == '个人服务工作者':
        return 2
    elif x == '（副）处级':
        return 3
    elif x == '中层管理人员':
        return 4
    elif x == '一般管理人员':
        return 5
    elif x == '一般员工':
        return 6
    else:
        return 0


def fun_marriage(x):
    if x == '初婚':
        return 1
    elif x == '离婚':
        return 2
    elif x == '未说明的婚姻状况':
        return 3
    elif x == '未婚':
        return 4
    elif x == '丧偶':
        return 5
    else:
        return 0



def fun_residence(x):
    if x == '无按揭自置':
        return 1
    elif x == '按揭自置':
        return 2
    elif x == '共有住宅':
        return 3
    elif x == '其他':
        return 4
    elif x == '租房':
        return 5
    elif x == '集体宿舍':
        return 6
    elif x == '亲属产权':
        return 7
    elif x == '未知':
        return 8
    else:
        return 0



def fun_edu(x):
    if x == '初中':
        return 1
    elif x == '高中':
        return 2
    elif x == '大学本科':
        return 3
    elif x == '大专':
        return 4
    elif x == '其他':
        return 5
    elif x == '中专':
        return 6
    elif x == '硕士':
        return 7
    elif x == '博士':
        return 8
    elif x == '小学及以下':
        return 9
    elif x == '职高':
        return 10
    else:
        return 0





def fun_hgst_lev(x):
    if x == '高净值客户（钻石1）':
        return 1
    elif x == '高净值客户（钻石2）':
        return 2
    elif x == '高净值客户（钻石3）':
        return 3
    elif x == '财富客户（高端）':
        return 4
    elif x == '财富客户（中高端)':
        return 5
    elif x == '中端客户':
        return 6
    elif x == '大众客户':
        return 7
    else:
        return 0


def fun_high_product(x):
    if x == '存款':
        return 1
    elif x == '理财':
        return 2
    elif x == '保险':
        return 3
    elif x == '代销理财':
        return 4
    elif x == '基金':
        return 5
    elif x == '信托':
        return 6
    else:
        return 0



def fun_hgst_ast_tp(x):
    if x == '活期存款':
        return 1
    elif x == '定期存款':
        return 2
    elif x == '理财':
        return 3
    elif x == '保险':
        return 4
    elif x == '代销理财':
        return 5
    elif x == '基金':
        return 6
    elif x == '信托':
        return 7
    else:
        return 0


def ch_name(x,file_path):
    df = pd.read_csv(file_path,sep = ',',header = None,names = ['Eng_name','Cha_name'])

    name_dict = dict(zip(df['Eng_name'],df['Cha_name']))

    output = []

    for name in x:
        if name in name_dict:
            output.append([name,name_dict[name]])
        else:
            output.append([name,name])

    result_df = pd.DataFrame(output,columns = ['Eng_name','Cha_name'])
    return result_df
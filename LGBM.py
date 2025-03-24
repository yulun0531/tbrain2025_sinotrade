import pandas as pd
import os
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, log_loss
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from collections import Counter

# 設定數據存放路徑
output_dir = r'C:\Users\yulun\Downloads\tbrain2025_sinotrade\Split_Files'
mi_scores_path = r'C:\Users\yulun\Downloads\mi_scores.csv'
feature_importance_path = r'C:\Users\yulun\Downloads\feature_importance.csv'

# 讀取並合併 CSV 文件
df_list = []
for i in range(1, 22):
    file_path = os.path.join(output_dir, f'data{i}.csv')
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df_list.append(df)
    print(f"第 {i} 個檔案載入成功")
merged_df = pd.concat(df_list, ignore_index=True)

# 檢查並填補 NaN 值
if merged_df.isnull().sum().sum() > 0:
    print("發現 NaN 值，開始填補...")
    merged_df.fillna(0, inplace=True)
    print("NaN 填補完成！")

# 移除非數值型欄位
for col in merged_df.select_dtypes(include=['object']).columns:
    merged_df.drop(columns=[col], inplace=True)

# 確認目標變數存在
if '飆股' not in merged_df.columns:
    raise ValueError("找不到目標變數 '飆股'，請確認資料集是否正確！")

# 分離特徵與目標變數
X = merged_df.drop('飆股', axis=1)
y = merged_df['飆股']

# 計算互信息
#print("計算互信息")
#mi_scores = mutual_info_classif(X, y, discrete_features=False)
#mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
#mi_df = mi_df.sort_values(by='MI_Score', ascending=False)

# 讀取 LightGBM 特徵重要性
feature_importance_df = pd.read_csv(feature_importance_path, encoding='utf-8-sig')
mi_df = pd.read_csv(mi_scores_path, encoding='utf-8-sig')
# 確保欄位名稱一致
feature_importance_df.rename(columns={'Feature': 'Feature', 'Importance': 'LGBM_Score'}, inplace=True)
mi_df.rename(columns={'Feature': 'Feature', 'MI_Score': 'MI_Score'}, inplace=True)
mi_df = mi_df.sort_values(by='MI_Score', ascending=False)

# 設定特徵重要性排名（數值越大代表越重要）
feature_importance_df['LGBM_Rank'] = feature_importance_df['LGBM_Score'].rank(ascending=False, method='average')
mi_df['MI_Rank'] = mi_df['MI_Score'].rank(ascending=False, method='average')

# 合併兩個 DataFrame
merged_df = pd.merge(feature_importance_df, mi_df, on='Feature', how='inner')

# 計算綜合排名（可以用平均或加權方式）
merged_df['Final_Rank'] = (merged_df['LGBM_Rank'] + merged_df['MI_Rank']) / 2  # 平均排名
# merged_df['Final_Rank'] = 0.7 * merged_df['LGBM_Rank'] + 0.3 * merged_df['MI_Rank']  # 加權排名（LightGBM 權重高）

# 根據排名排序
merged_df = merged_df.sort_values(by='Final_Rank', ascending=True)

# 選擇前 top_k_features 個特徵
top_k_features = 100  # 可根據需求調整
selected_features = merged_df.iloc[:top_k_features]['Feature'].tolist()
X = X[selected_features]

# 轉換為 numpy array
X = X.to_numpy()
y = y.to_numpy()

# 計算正負樣本數量
num_pos = np.sum(y == 1)
num_neg = np.sum(y == 0)
scale_pos_weight = num_neg / num_pos

# 設定交叉驗證
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 訓練參數
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 50,
    'max_depth': 6,
    'max_bin': 511,
    'learning_rate': 0.06,
    'feature_fraction': 0.8,
    'feature_fraction_seed': 42,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'min_data_in_leaf': 60,
    'lambda_L1': 0.6,
    'lambda_L2': 0.6,
    'scale_pos_weight': scale_pos_weight,
    'verbose': -1,
}

# 初始化評估指標
auc_scores, accuracy_scores, f1_scores, precision_scores, recall_scores, models = [], [], [], [], [], []

# 進行 K-Fold 訓練
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n🚀 正在訓練第 {fold+1} 折...")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    gbm = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=5)
        ]
    )

    # 預測與評估
    y_pred = gbm.predict(X_val)
    y_pred_binary = (y_pred > 0.3).astype(int)

    auc = roc_auc_score(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred_binary)
    f1 = f1_score(y_val, y_pred_binary)
    precision = precision_score(y_val, y_pred_binary)
    recall = recall_score(y_val, y_pred_binary)

    print(f"✅ 第 {fold+1} 折結果 - AUC: {auc:.4f}, 準確率: {accuracy:.4f}, F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    auc_scores.append(auc)
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)
    models.append(gbm)

# **🔟 顯示最終平均結果**
print("\n📊 K-Fold 交叉驗證結果 (平均值):")
print(f"🎯 平均 AUC: {sum(auc_scores) / n_splits:.4f}")
print(f"🎯 平均 準確率: {sum(accuracy_scores) / n_splits:.4f}")
print(f"🎯 平均 F1-score: {sum(f1_scores) / n_splits:.4f}")
print(f"🎯 平均 Precision: {sum(precision_scores) / n_splits:.4f}")
print(f"🎯 平均 Recall: {sum(recall_scores) / n_splits:.4f}")

# **🏆 檢測過擬合**
if min(auc_scores) < 0.85:  # 若某折驗證 AUC 過低
    print("⚠️ 可能過擬合！建議減少 `num_leaves` 或增加 `feature_fraction`。")
# 儲存最佳模型
best_model_index = np.argmax(auc_scores)
best_model = models[best_model_index]
model_path = r'C:\Users\yulun\Downloads\lightgbm_best_model.pkl'
joblib.dump(best_model, model_path)
print(f"最佳模型已儲存至 {model_path}")

# 儲存選定特徵
selected_features_path = r'C:\Users\yulun\Downloads\selected_features.txt'
np.savetxt(selected_features_path, selected_features, fmt='%s', encoding='utf-8')
print(f"選定的特徵已儲存至 {selected_features_path}")

# 儲存互信息數值
mi_scores_path = r'C:\Users\yulun\Downloads\mi_scores.csv'
mi_df.to_csv(mi_scores_path, index=False, encoding='utf-8-sig')
print(f"互信息數據已儲存至 {mi_scores_path}")
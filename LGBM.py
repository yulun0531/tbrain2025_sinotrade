import pandas as pd
import os
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, log_loss
from imblearn.over_sampling import SMOTE
from collections import Counter

# 設定數據存放路徑
output_dir = r'C:\Users\yulun\Downloads\38_Training_Data_Set\Split_Files'

# 讀取並合併 CSV 文件
df_list = []
for i in range(1, 6):
    file_path = os.path.join(output_dir, f'data{i}.csv')
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df_list.append(df)
    print(f"第 {i} 個檔案載入成功")
merged_df = pd.concat(df_list, ignore_index=True)

# 檢查並填補 NaN 值
if merged_df.isnull().sum().sum() > 0:
    print("發現 NaN 值，開始填補...")
    print(f"🛑 測試資料有NAN，缺少: {merged_df.isnull().sum().sum()}個")
    merged_df.fillna(0, inplace=True)
    print("NaN 填補完成！")

# 移除非數值型欄位
for col in merged_df.select_dtypes(include=['object']).columns:
    print(f"移除非數值欄位: {col}")
    merged_df.drop(columns=[col], inplace=True)

# 確認目標變數存在
if '飆股' not in merged_df.columns:
    raise ValueError("找不到目標變數 '飆股'，請確認資料集是否正確！")

# 分離特徵與目標變數
X = merged_df.drop('飆股', axis=1)
y = merged_df['飆股']

# 轉換為 numpy array
X = X.to_numpy()
y = y.to_numpy()

# 計算正負樣本數量
num_pos = np.sum(y == 1)
num_neg = np.sum(y == 0)
scale_pos_weight = num_neg / num_pos
print(f"正樣本數量: {num_pos}, 負樣本數量: {num_neg}, 計算得到的 scale_pos_weight: {scale_pos_weight:.4f}")

# 設定交叉驗證
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# **6️⃣ 訓練模型**
params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,  #因為特徵較複雜嘗試提高
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'scale_pos_weight': scale_pos_weight,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'verbose': -1
    }
## 撰寫訓練用的參數
params = {
        'task': 'train',
        ## 目標函數
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        ## 算法類型
        'num_leaves': 50,
        'max_depth': 6,
        'max_bin' : 511,
        'learning_rate': 0.06,
        ## 構建樹時的特徵選擇比例
        'feature_fraction': 0.8,
        'feature_fraction_seed': 42,
        "bagging_fraction":0.8,
        ## k 表示每k次迭代就進行bagging
        'bagging_freq':5,
        "min_child_samples": 20,
        'min_data_in_leaf': 60,
        'lambda_L1': 0.6,
        'lambda_L2': 0.6,
        ## 如果數據集樣本分布不均衡，可以幫助明顯提高準確率
        'scale_pos_weight': scale_pos_weight,
        'verbose':-1,
}
auc_scores = []
accuracy_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
train_log_losses = []
val_log_losses = []
models = []

# **3️⃣ 進行 K-Fold 訓練**
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\n🚀 正在訓練第 {fold+1} 折...")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print(f"🎯 訓練集中各類別分布: {Counter(y_train)}")

    # **5️⃣ 準備 LGBM 數據集**
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    gbm = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=25),
            lgb.log_evaluation(period=5)
        ]
    )

    # **7️⃣ 預測與評估**
    y_pred_train = gbm.predict(X_train)
    y_pred = gbm.predict(X_val)

    y_pred_train_binary = (y_pred_train > 0.3).astype(int)
    y_pred_binary = (y_pred > 0.3).astype(int)

    # **8️⃣ 計算評估指標**
    train_log_loss = log_loss(y_train, y_pred_train)
    val_log_loss = log_loss(y_val, y_pred)

    auc = roc_auc_score(y_val, y_pred)
    accuracy = accuracy_score(y_val, y_pred_binary)
    f1 = f1_score(y_val, y_pred_binary)
    precision = precision_score(y_val, y_pred_binary)
    recall = recall_score(y_val, y_pred_binary)
    print(f"預測機率範圍: {np.min(y_pred):.4f} ~ {np.max(y_pred):.4f}")
    print(f"預測大於 0.3 的比例: {np.mean(y_pred > 0.3):.4f}")

    print(f"✅ 第 {fold+1} 折結果 - AUC: {auc:.4f}, 準確率: {accuracy:.4f}, F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"🛠️ 訓練 Log Loss: {train_log_loss:.4f}, 驗證 Log Loss: {val_log_loss:.4f}")
    
    # **9️⃣ 儲存評估結果**
    auc_scores.append(auc)
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)
    train_log_losses.append(train_log_loss)
    val_log_losses.append(val_log_loss)
    models.append(gbm)

# **🔟 顯示最終平均結果**
print("\n📊 K-Fold 交叉驗證結果 (平均值):")
print(f"🎯 平均 AUC: {sum(auc_scores) / n_splits:.4f}")
print(f"🎯 平均 準確率: {sum(accuracy_scores) / n_splits:.4f}")
print(f"🎯 平均 F1-score: {sum(f1_scores) / n_splits:.4f}")
print(f"🎯 平均 Precision: {sum(precision_scores) / n_splits:.4f}")
print(f"🎯 平均 Recall: {sum(recall_scores) / n_splits:.4f}")
print(f"🛠️ 平均 訓練 Log Loss: {sum(train_log_losses) / n_splits:.4f}")
print(f"🛠️ 平均 驗證 Log Loss: {sum(val_log_losses) / n_splits:.4f}")

# **🏆 檢測過擬合**
if min(auc_scores) < 0.85:  # 若某折驗證 AUC 過低
    print("⚠️ 可能過擬合！建議減少 `num_leaves` 或增加 `feature_fraction`。")

best_model_index = np.argmax(auc_scores)
best_model = models[best_model_index]
model_path = r'C:\Users\yulun\Downloads\lightgbm_best_model.pkl'
joblib.dump(best_model, model_path)
print(f"最佳模型已儲存至 {model_path}")

feature_names_path = r'C:\Users\yulun\Downloads\feature_names.txt'
np.savetxt(feature_names_path, merged_df.columns[:-1], fmt='%s', encoding='utf-8')
print(f"特徵名稱已儲存至 {feature_names_path}")

importance_path = r'C:\Users\yulun\Downloads\feature_importance.csv'
pd.DataFrame({
    'Feature': merged_df.columns[:-1],
    'Importance': best_model.feature_importance(importance_type='gain')
}).sort_values(by='Importance', ascending=False).to_csv(importance_path, index=False, encoding='utf-8-sig')
print(f"特徵重要性已儲存至 {importance_path}")
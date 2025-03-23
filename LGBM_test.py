import pandas as pd
import lightgbm as lgb
import joblib
import os

# **檔案路徑**
test_file_path = r'C:\Users\yulun\Desktop\38_Public_Test_Set_and_Submmision_Template\38_Public_Test_Set_and_Submmision_Template\public_x.csv'
submission_template_path = r'C:\Users\yulun\Desktop\38_Public_Test_Set_and_Submmision_Template\38_Public_Test_Set_and_Submmision_Template\submission_template_public.csv'
model_path = r'C:\Users\yulun\Downloads\lightgbm_best_model.pkl'
feature_names_path = r'C:\Users\yulun\Downloads\feature_names.txt'  # 訓練時的特徵名稱存放路徑

# **1️⃣ 讀取測試資料**
print("📥 讀取測試資料...")
test_df = pd.read_csv(test_file_path, encoding="utf-8-sig")

# **2️⃣ 讀取 LightGBM 模型**
print("📦 載入 LightGBM 模型...")
gbm = joblib.load(model_path)

# **3️⃣ 讀取訓練時的特徵名稱**
print("📜 載入訓練特徵名稱...")
with open(feature_names_path, 'r', encoding='utf-8-sig') as f:
    feature_names = [line.strip() for line in f.readlines()]

# **4️⃣ 處理測試資料特徵對齊**
print("🔄 對齊測試資料的特徵...")

# **(a) 移除測試資料中多餘的特徵**
extra_cols = [col for col in test_df.columns if col not in feature_names]
if extra_cols:
    print(f"🛑 測試資料有 {len(extra_cols)} 個多餘特徵，刪除: {extra_cols}")
    test_df = test_df.drop(columns=extra_cols)

# **(b) 補齊測試資料缺少的特徵**
missing_cols = [col for col in feature_names if col not in test_df.columns]
print(f"🛑 測試資料有個缺少特徵，缺少: {missing_cols}")
if missing_cols:
    print(f"⚠️ 測試資料缺少 {len(missing_cols)} 個特徵，補齊中: {missing_cols}")
    for col in missing_cols:
        test_df[col] = 0  # 填充為 0

# **(c) 確保測試資料的特徵順序與訓練時相同**
test_df = test_df[feature_names]

# **(d) 處理測試資料中的 NaN**
print("⚠️ 檢查並填補測試資料中的 NaN 值（先前向填充，再後向填充）...")
test_df.fillna(0, inplace=True)

# **5️⃣ 開始預測**
print("📊 開始預測...")
y_pred = gbm.predict(test_df, num_iteration=gbm.best_iteration)

# **6️⃣ 轉換為二元分類結果**
y_pred_binary = [1 if pred > 0.3 else 0 for pred in y_pred]

# **7️⃣ 讀取提交模板**
print("📄 讀取提交模板...")
submission_df = pd.read_csv(submission_template_path, encoding="utf-8-sig")

# 確保提交模板中包含 'ID' 列
if 'ID' not in submission_df.columns:
    raise ValueError("❌ 提交模板中缺少 'ID' 列，請檢查提交檔案格式！")

# **8️⃣ 儲存預測結果**
submission_df['飆股'] = y_pred_binary

# **9️⃣ 儲存提交檔案**
submission_file_path = r'C:\Users\yulun\Desktop\submission.csv'
os.makedirs(os.path.dirname(submission_file_path), exist_ok=True)
submission_df.to_csv(submission_file_path, index=False, encoding='utf-8', lineterminator='\n')

print(f"✅ 提交文件已成功保存至 {submission_file_path}")

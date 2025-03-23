# LightGBM 修改過程與結果記錄

## 1. 專案概述
本文件記錄了對 LightGBM (LGBM) 進行的修改，包含修改的內容、目的、影響，以及最終的結果。

## 2. 初始版本

### 2.1 環境設置
- **語言與套件**: Python, pandas, os, scikit-learn, LightGBM
- **GPU 設定**: 透過 `device: gpu` 啟用 GPU 加速

### 2.2 數據處理
1. **數據讀取與合併**
   - 讀取 `data1.csv` ~ `data21.csv`，合併為單一 DataFrame。
2. **數據前處理**
   - 填補數值型缺失值 (以0填補)。

### 2.3 訓練流程
1. **數據集切分**: 80% 訓練集, 20% 測試集。
2. **LightGBM 參數設置**:
   - `boosting_type: gbdt`
   - `objective: binary`
   - `metric: binary_logloss`
   - `device: gpu`
3. **模型訓練**:
   - 使用 `lgb.Dataset` 建立數據集。
   - 設置 `early_stopping` (10 輪) 及 `log_evaluation` (每 10 輪顯示)。
   - 訓練 100 輪。
4. **模型保存**:
   - 訓練完成後保存模型 (`lightgbm_model.txt`)。
   - 保存特徵名稱 (`feature_names.txt`)。
5. **預測與評估**:
   - 預測測試集，使用 `0.5` 為閾值轉換為二元標籤。
   - 計算 `accuracy_score` 和 `f1_score`。

## 3. 修改過程

### (1) 第一版 ➝ 第二版
- **超參數調整**:
  - `learning_rate`: 減少學習率，避免過擬合。
  - `num_leaves`: 增加葉子數量，提高模型擬合能力。
  - `max_depth`: 限制樹的深度，防止過度擬合。
  - `min_data_in_leaf`: 增加葉節點的最少數據量，提升泛化能力。
- **Loss Function 修改**
  - 改用 `huber loss` 來減少異常值影響。
- **Boosting 方式比較**
  - `gbdt` (預設)
  - `dart` (Dropout 方式)
  - `goss` (Gradient-based One-Side Sampling)

### (2) 第二版 ➝ 第三版
- **交叉驗證改進**:
  - 使用 **StratifiedKFold (5-Fold)**，避免資料分布不均影響模型學習。
- **類別不均衡處理**:
  - 設定 `scale_pos_weight` 動態調整類別權重。
  - 啟用 `is_unbalance: True` 讓 LightGBM 自動調整類別比重。
- **閾值調整**:
  - 原預設 `0.5` 改為 `0.4`，提升 **Recall**。
- **特徵重要性分析**:
  - 計算 `gain-based feature importance`，過濾低影響特徵。

---

## 4. 結果比較

### 4.1 第一版 ➝ 第二版
| 指標 | 第一版結果 | 第二版結果 |
|------|------|------|
| **Advanced_Public_Precision** | 0.2468 | 0.6321 |
| **Advanced_Public_Recall** | 0.2216 | 0.3807 |

### 4.2 第二版 ➝ 第三版
| 指標 | 第二版結果 | 第三版結果 |
|------|------|------|
| **Advanced_Public_Precision** | 0.6321 | 0.7481 |
| **Advanced_Public_Recall** | 0.3807 | 0.5568 |

---

## 5. 結論
透過三個版本的優化，我們的模型在 **Advanced_Public_Precision** 上從 **0.2468** 提升到 **0.7481**，**Advanced_Public_Recall** 也從 **0.2216** 提升到 **0.5568**，顯示模型預測準確率及召回率均有明顯提升。

未來可以進一步：
1. **調整超參數**: 探索更適合的 `num_leaves`、`max_depth` 等數值。
2. **更進階的特徵工程**: 測試 PCA 降維，或引入更多交互特徵。
3. **調整閾值策略**: 讓 Precision & Recall 達到更理想的平衡。

---

## 6. 參考資料
- LightGBM 官方文件: [https://lightgbm.readthedocs.io](https://lightgbm.readthedocs.io)
- 相關技術論文與文章

# 🚀 クイックスタートガイド

このガイドでは、競馬予測AIシステムを最速で立ち上げて動かす手順を説明します。

## 前提条件

- Python 3.9以上がインストールされていること
- pipが利用可能であること

## セットアップ（5分）

### 1. リポジトリのクローン

```bash
git clone https://github.com/AS-t-u-m-s-a-t/Keiba-predictor.git
cd Keiba-predictor
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
pip install -e .
```

## 動作確認（3分）

### 1. デモスクリプトを実行

```bash
python example.py
```

このスクリプトは以下を実行します：
- サンプルデータの生成
- データ前処理
- 特徴量生成
- モデル学習
- 予測と評価

実行結果：
```
============================================================
競馬予測AIシステム - デモンストレーション
============================================================
=== サンプルデータ作成 ===
✓ サンプルデータ作成完了: 200件
...
✓ 学習完了
✓ 評価完了
  MAE (平均絶対誤差): 2.530
  RMSE (二乗平均平方根誤差): 2.892
============================================================
デモンストレーション完了！
============================================================
```

### 2. テストを実行

```bash
pytest tests/ -v
```

期待される結果：
```
27 passed, 5 warnings in 1.58s
```

## Web UIの起動（1分）

```bash
streamlit run src/app/streamlit_app.py
```

ブラウザが自動的に開き、`http://localhost:8501` でダッシュボードが表示されます。

### Web UIの使い方

1. **データアップロード**
   - 左側のファイルアップローダーからCSVをアップロード
   - サンプルデータ: `data/raw/sample_races.csv`

2. **前処理を実行**
   - 「前処理を実行」ボタンをクリック

3. **予測を実行**（学習済みモデルがある場合）
   - 「予測を実行」ボタンをクリック
   - レース選択で結果を確認

## 実データでの利用

### データスクレイピング

```python
from src.scraper.netkeiba import NetkeibaScraper

# スクレイパー初期化
scraper = NetkeibaScraper(delay=1.0, output_dir="data/raw")

# レースIDを生成（例：2023年5月の東京競馬場）
race_ids = scraper.generate_race_ids(
    year=2023,
    month=5,
    track_code="05",  # 東京
    day=5,
    race_num=12
)

# データ収集
data = scraper.scrape_races(race_ids, output_file="races_2023_05.csv")
```

### データ処理とモデル学習

```python
import pandas as pd
from src.preprocessing.cleaner import DataCleaner
from src.features.engineer import FeatureEngineer
from src.models.lightgbm_model import LightGBMModel

# データ読み込み
df = pd.read_csv("data/raw/races_2023_05.csv")

# 前処理
cleaner = DataCleaner()
df_cleaned = cleaner.clean(df)

# 特徴量生成
engineer = FeatureEngineer()
df_featured = engineer.create_features(df_cleaned)

# モデル学習
feature_names = engineer.get_feature_names(df_featured)
X = df_featured[feature_names]
y = df_featured['finish_position']

model = LightGBMModel(random_state=42)
model.train(X, y, n_splits=5, num_boost_round=1000)

# モデル保存
model.save("models/my_model.joblib")
```

### 予測

```python
# 新しいデータで予測
predictions = model.predict(X_new)
win_proba = model.predict_proba_win(X_new)
```

## トラブルシューティング

### ImportError: No module named 'xxx'

```bash
pip install -r requirements.txt
```

### LightGBM installation error

LightGBMのインストールに失敗する場合：

```bash
# macOS
brew install cmake libomp
pip install lightgbm

# Ubuntu/Debian
sudo apt-get install cmake
pip install lightgbm

# Windows
# Visual C++ Build Toolsが必要
pip install lightgbm
```

### Streamlitが起動しない

```bash
# Streamlitを再インストール
pip install --upgrade streamlit

# ポート8501が使用中の場合、別のポートを指定
streamlit run src/app/streamlit_app.py --server.port 8502
```

## 次のステップ

### より高度な使い方

1. **Jupyter Notebookで探索的データ分析**
   ```bash
   jupyter notebook notebooks/exploration.ipynb
   ```

2. **ハイパーパラメータチューニング**
   - Optunaなどを使用した自動チューニング
   - カスタムパラメータでモデル作成

3. **モデルの追加**
   - `src/models/`に新しいモデルを追加
   - `BaseModel`を継承して実装

### ドキュメント

- **README.md**: 詳細なドキュメント
- **IMPLEMENTATION_SUMMARY.md**: 実装サマリー
- **ソースコード**: 各ファイルのdocstringを参照

## サポート

問題が発生した場合は、GitHubのIssueで報告してください：
https://github.com/AS-t-u-m-s-a-t/Keiba-predictor/issues

---

**Happy Horse Racing Prediction! 🏇**

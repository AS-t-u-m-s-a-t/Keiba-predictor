# 🏇 競馬予測AIシステム

フルスタックで構築された競馬予測AIシステムです。データスクレイピングから機械学習モデルの学習、Web UIまでを含む包括的なソリューションを提供します。

## 📋 目次

- [概要](#概要)
- [主な機能](#主な機能)
- [プロジェクト構成](#プロジェクト構成)
- [技術スタック](#技術スタック)
- [セットアップ](#セットアップ)
- [使い方](#使い方)
- [モデルについて](#モデルについて)
- [将来の拡張計画](#将来の拡張計画)
- [ライセンス](#ライセンス)

## 概要

このシステムは、競馬レースの予測を行うためのAIシステムです。netkeiba.comからデータをスクレイピングし、機械学習モデル（LightGBM）を使用して各馬の着順を予測します。

### 主な特徴

- 🔍 **自動データ収集**: netkeiba.comから自動でレースデータを収集
- 🧹 **データ前処理**: 欠損値処理、型変換、異常値除去を自動実行
- ⚙️ **特徴量エンジニアリング**: 過去成績、騎手勝率、距離適性など多様な特徴量を生成
- 🤖 **高精度予測モデル**: LightGBMベースの予測モデル（時系列クロスバリデーション対応）
- 📊 **詳細な評価**: 的中率、回収率、Top-N精度など多角的な評価
- 🖥️ **Web UI**: Streamlitベースの直感的なダッシュボード
- 🔧 **拡張性**: Strategyパターンで新しいモデルを簡単に追加可能

## 主な機能

### 1. データスクレイピング
- netkeiba.comからレース結果を自動収集
- レート制限機能でサーバーに負荷をかけない設計
- レース情報（競馬場、距離、馬場状態、天候）の取得
- 馬情報（着順、馬名、騎手、タイム、オッズなど）の取得

### 2. データ前処理
- 欠損値の適切な処理
- データ型の自動変換（タイム→秒数など）
- 異常値の検出と除去
- カテゴリ変数のエンコーディング

### 3. 特徴量エンジニアリング
- **馬の過去成績**: 過去N走の平均着順、前走着順、勝率、連対率
- **騎手の実績**: 勝率、連対率、複勝率、平均着順
- **距離適性**: 同距離カテゴリでの過去平均着順
- **枠番の有利不利**: 枠番ごとの統計情報、内枠/外枠フラグ
- **コース適性**: 芝/ダート別の過去成績

### 4. 予測モデル
- **LightGBM**（実装済み）: メインの予測モデル
  - 時系列クロスバリデーション対応
  - ハイパーパラメータチューニング
  - 特徴量重要度の可視化
- **ニューラルネットワーク**（プレースホルダー）: 将来実装予定
- **LambdaMART**（プレースホルダー）: 将来実装予定

### 5. 評価機能
- 的中率（1着予測の正解率）
- Top-N精度（上位N頭を当てられた割合）
- 回収率シミュレーション（単勝・複勝）
- 混同行列の可視化
- 着順ごとの予測精度

### 6. Web UI
- Streamlitベースのダッシュボード
- レース予測結果の表示
- 特徴量重要度のグラフ表示
- モデル選択機能
- CSVデータのアップロード機能

## プロジェクト構成

```
keiba-ai/
├── src/
│   ├── __init__.py
│   ├── scraper/              # データスクレイピング
│   │   ├── __init__.py
│   │   ├── netkeiba.py       # netkeiba.comからのデータ収集
│   │   └── parser.py         # HTMLパース処理
│   ├── preprocessing/        # データ前処理
│   │   ├── __init__.py
│   │   └── cleaner.py        # 欠損値処理、型変換、クリーニング
│   ├── features/             # 特徴量エンジニアリング
│   │   ├── __init__.py
│   │   └── engineer.py       # 特徴量作成
│   ├── models/               # 予測モデル（Strategyパターン）
│   │   ├── __init__.py
│   │   ├── base.py           # モデル基底クラス
│   │   ├── lightgbm_model.py # LightGBMモデル（実装済み）
│   │   ├── neural_net.py     # ニューラルネット（プレースホルダー）
│   │   └── lambdamart.py     # LambdaMART（プレースホルダー）
│   ├── evaluation/           # 評価
│   │   ├── __init__.py
│   │   └── evaluator.py      # 的中率、回収率シミュレーション
│   └── app/                  # Web UI
│       ├── __init__.py
│       └── streamlit_app.py  # Streamlit ダッシュボード
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
├── data/
│   ├── raw/                  # 生データ
│   └── processed/            # 前処理済みデータ
├── notebooks/
│   └── exploration.ipynb     # データ分析用Notebook
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## 技術スタック

### 言語・フレームワーク
- **Python 3.9+**: メイン言語
- **LightGBM**: 機械学習モデル
- **Streamlit**: Web UI

### データ処理
- **pandas**: データ操作
- **numpy**: 数値計算
- **scikit-learn**: 機械学習ツール

### スクレイピング
- **BeautifulSoup4**: HTMLパース
- **requests**: HTTP通信

### 可視化
- **matplotlib**: グラフ描画
- **plotly**: インタラクティブグラフ
- **seaborn**: 統計的可視化

### テスト
- **pytest**: ユニットテスト

## セットアップ

### 必要な環境
- Python 3.9以上
- pip

### インストール手順

1. **リポジトリのクローン**
```bash
git clone https://github.com/AS-t-u-m-s-a-t/Keiba-predictor.git
cd Keiba-predictor
```

2. **仮想環境の作成（推奨）**
```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
```

3. **依存パッケージのインストール**
```bash
pip install -r requirements.txt
```

4. **パッケージのインストール（開発モード）**
```bash
pip install -e .
```

## 使い方

### 1. データスクレイピング

```python
from src.scraper.netkeiba import NetkeibaScraper

# スクレイパーのインスタンス化
scraper = NetkeibaScraper(delay=1.0, output_dir="data/raw")

# レースIDを生成（例: 2023年5月、東京競馬場）
race_ids = scraper.generate_race_ids(
    year=2023, 
    month=5, 
    track_code="05",  # 東京
    day=5,            # 5日間
    race_num=12       # 各日12レース
)

# データを収集
data = scraper.scrape_races(race_ids, output_file="races_2023_05.csv")
```

### 2. データ前処理と特徴量生成

```python
import pandas as pd
from src.preprocessing.cleaner import DataCleaner
from src.features.engineer import FeatureEngineer

# データ読み込み
df = pd.read_csv("data/raw/races_2023_05.csv")

# 前処理
cleaner = DataCleaner()
df_cleaned = cleaner.clean(df)

# 特徴量生成
engineer = FeatureEngineer()
df_featured = engineer.create_features(df_cleaned)

# 保存
df_featured.to_csv("data/processed/featured_data.csv", index=False)
```

### 3. モデル学習

```python
from src.models.lightgbm_model import LightGBMModel

# データ準備
feature_names = engineer.get_feature_names(df_featured)
X = df_featured[feature_names]
y = df_featured['finish_position']

# モデル学習
model = LightGBMModel(random_state=42)
model.train(X, y, n_splits=5, num_boost_round=1000)

# モデル保存
model.save("models/lightgbm_model.joblib")
```

### 4. 予測

```python
# 予測
predictions = model.predict(X)

# 勝率予測
win_proba = model.predict_proba_win(X)
```

### 5. 評価

```python
from src.evaluation.evaluator import Evaluator

evaluator = Evaluator()

# 総合評価
metrics = evaluator.evaluate(
    y_true=y,
    y_pred=predictions,
    race_ids=df_featured['race_id'].values,
    odds=df_featured['odds'].values
)

print(f"MAE: {metrics['mae']:.3f}")
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"Top-3精度: {metrics['top3_accuracy']:.2%}")
print(f"単勝回収率: {metrics['win_payback_rate']:.1f}%")
```

### 6. Web UI起動

```bash
streamlit run src/app/streamlit_app.py
```

ブラウザで `http://localhost:8501` にアクセスしてダッシュボードを表示。

## モデルについて

### Strategyパターンの採用

このシステムでは、モデルをStrategyパターンで設計しています。これにより、異なる予測モデルを統一インターフェースで扱うことができ、新しいモデルの追加が容易になります。

#### 基底クラス（BaseModel）

すべてのモデルは `BaseModel` 抽象クラスを継承します：

```python
class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y, **kwargs): pass
    
    @abstractmethod
    def predict(self, X): pass
    
    @abstractmethod
    def save(self, filepath): pass
    
    @abstractmethod
    def load(self, filepath): pass
```

#### 実装済みモデル

**LightGBMModel**: 勾配ブースティング決定木を使用した回帰モデル
- 時系列クロスバリデーション対応
- 特徴量重要度の可視化
- ハイパーパラメータチューニング対応

#### 将来実装予定のモデル

**NeuralNetModel**: LSTM/Transformerベースのニューラルネットワーク
- 時系列データの活用
- Embeddingレイヤーでカテゴリ変数を処理

**LambdaMARTModel**: ランキング学習アルゴリズム
- レース内での順位予測に特化
- NDCG最適化

### 新しいモデルの追加方法

1. `src/models/` に新しいモデルファイルを作成
2. `BaseModel` を継承
3. 必須メソッド（train, predict, save, load）を実装
4. Web UIのモデル選択に追加

## テスト

```bash
# すべてのテストを実行
pytest

# カバレッジ付きで実行
pytest --cov=src tests/

# 特定のテストファイルのみ実行
pytest tests/test_models.py
```

## 設計方針

### 拡張性
- Strategyパターンでモデルを簡単に追加可能
- モジュール化された設計で保守性を確保

### 再現性
- 乱数シードの固定
- データパイプラインの整備
- モデルの保存・読み込み機能

### リーク防止
- 時系列を厳密に管理
- 未来の情報を使わないよう特徴量生成を工夫
- TimeSeriesSplitを使用したクロスバリデーション

### コード品質
- 型ヒントの使用
- docstringによる詳細なドキュメント
- 適切なエラーハンドリング
- ユニットテストによる品質保証

## 将来の拡張計画

### 短期
- [ ] より多くの特徴量の追加（調教師の実績、血統情報など）
- [ ] ハイパーパラメータの自動チューニング（Optuna）
- [ ] アンサンブルモデルの実装

### 中期
- [ ] ニューラルネットワークモデルの実装
- [ ] LambdaMARTランキングモデルの実装
- [ ] リアルタイム予測機能

### 長期
- [ ] ディープラーニングモデルの高度化
- [ ] 外部APIとの連携
- [ ] モバイルアプリの開発

## 注意事項

- **データスクレイピング**: netkeiba.comの利用規約を遵守してください。商用利用や過度なリクエストは避けてください。
- **予測精度**: このシステムは教育・研究目的です。実際の投資判断には慎重な検討が必要です。
- **ギャンブル依存**: 競馬は娯楽として楽しみましょう。

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します！大きな変更の場合は、まずIssueを開いて変更内容を議論してください。

## サポート

問題が発生した場合は、GitHubのIssueセクションで報告してください。

---

**Enjoy Horse Racing AI! 🏇**
# 実行順序

すべてのコマンドは `final_submission/` 直下で実行する。既存結果の閲覧だけなら工程1から9を再実行する必要はない。

| 順序 | 工程 | 入力 | 主な出力 | 外部接続 | 再現性・主要パラメータ |
|---:|---|---|---|---|---|
| 1 | `python code/pipeline/01_extract_text.py` | `data/raw/*.html` | `data/extracted/*.txt` | 不要 | 同一HTMLとBeautifulSoup環境では決定的。本文候補を選び、段落境界を保持する。 |
| 2 | `python code/pipeline/02_clean_text.py` | `data/extracted/*.txt` | `data/cleaned/*_clean.txt` | 不要 | 決定的。NFKC、空白・ボイラープレート処理、声明の参考資料部分を切除する。 |
| 3 | `python code/pipeline/03_create_chunks.py` | クリーニング済み本文、`data/metadata/documents.csv` | `data/cleaned/chunks.csv` | 不要 | 決定的。段落優先、目安150–400字、上限目安500字、文中分割なし。 |
| 4 | `python code/pipeline/04_run_tfidf.py` | `chunks.csv`、`stopwords_ja.txt`、メタデータ | `results/tfidf/*.csv`、補足図 | 不要 | 同一ライブラリでは決定的。Sudachi SplitMode.C、名詞・動詞・形容詞、unigram、`min_df=2`、`max_df=0.85`、sublinear TF、L2。 |
| 5 | `python code/pipeline/05_run_tfidf_nouns.py` | `chunks.csv`、`stopwords_ja_nouns.txt` | `results/tfidf_nouns/*.csv`、名詞のみ図 | 不要 | 同一ライブラリでは決定的。Sudachi SplitMode.C、名詞のみ、unigram、`min_df=2`、`max_df=0.90`、sublinear TF、L2。 |
| 6 | `python code/pipeline/06_run_embeddings.py` | `chunks.csv` | `results/embedding/*.csv`、クラスタ図 | 初回モデル取得時のみ必要 | モデルは `paraphrase-multilingual-MiniLM-L12-v2`。正規化埋め込み、KMeans `k=3,4,5`、`random_state=42`、`n_init=20`。環境差で浮動小数点に微差が出る場合がある。 |
| 7 | `python code/pipeline/07_prepare_llm_batches.py` | `chunks.csv` | `data/llm_batches/batch_*.jsonl`、分類テンプレート、固定プロンプト | 不要 | 決定的。元順序を保持し、1バッチ15チャンク。 |
| 8 | 外部LLM分類 | 固定プロンプト、JSONLバッチ | `results/llm/llm_labels.csv` 相当 | 外部AIアクセスが必要 | 非決定的になり得る。本文のみを根拠にJSON分類し、短い引用を付す。既存分類出力は収録済み。 |
| 9 | `python code/pipeline/08_analyze_llm_labels.py` | `chunks.csv`、`results/llm/llm_labels.csv` | `results/llm/` の集計CSV、補足図 | 不要 | 与えられた分類結果に対して決定的。文書均等化集計を主要比較に使用する。 |
| 10 | `python code/visualizations.py` | パッケージ内の結果CSV、クラスタ名対応表 | `figures/presentation/`、`figures/supplementary/` | 不要 | 既存分析結果だけを可視化する。TF-IDF再学習、埋め込み生成、KMeans、LLM分類は行わない。 |

外部LLM分類の出力は `results/llm/` に既に含まれているため、結果の確認と図の再生成には工程8を再実行しなくてよい。工程10は任意入力が欠けている図だけを警告付きでスキップし、他の図を継続する。

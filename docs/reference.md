# vJEPA2 model + mask 統合コピー（scriptsエントリ版, 2026-04-27）

プロジェクトrootで `scripts/` を実行エントリ、`src/` をライブラリ層として分離した構成です。

## 構成
- `src/models/` : ViT本体・predictor・model utils
- `src/masks/` : mask生成本体・mask適用
- `src/utils/` : 分散・logging・scheduler・checkpoint loader を含む学習ユーティリティ
- `scripts/train/` : stage1 学習エントリと学習本体
- `scripts/mae/` : MAE 事前学習エントリと学習本体
- `scripts/downstream/` : downstream 学習エントリと学習本体

## メモ
- `source/src/...` の二重構造は廃止
- model内部 import は `src.models...` を向くように調整済み
- stage1 用イベントデータセットは `src/datasets/` に追加（詳細: `docs/stage1_event_dataset.md`）
- stage1 学習エントリは `scripts/train/run_train.py`（詳細: `docs/stage1_training.md`）
- MAE 学習エントリは `scripts/mae/run_mae.py`（詳細: `docs/mae_pretraining.md`）
- downstream 学習エントリは `scripts/downstream/run_downstream.py`（詳細: `docs/downstream_training.md`）
- Hydra config には `img_data` / `img_mask` グループを追加済み（`img_data=event_h5_single_frame` で 1時刻サンプル branch を有効化可能）

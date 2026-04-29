# vJEPA2 model + mask 統合コピー（src起点版, 2026-04-27）

プロジェクトrootから `src` を起点に分岐する構成に整理しています。

## 構成
- `src/models/` : ViT本体・predictor・model utils
- `src/masks/` : mask生成本体・mask適用
- `src/utils/` : 分散・logging・scheduler・checkpoint loader を含む学習ユーティリティ
- `src/training/` : vjepa2.1 参照の stage1 学習ロジック

## メモ
- `source/src/...` の二重構造は廃止
- model内部 import は `src.models...` を向くように調整済み
- stage1 用イベントデータセットは `src/datasets/` に追加（詳細: `docs/stage1_event_dataset.md`）
- stage1 学習エントリは `scripts/train/run_train.py`（詳細: `docs/stage1_training.md`）
- Hydra config には `img_data` / `img_mask` グループを追加済み（`img_data=event_h5_single_frame` で 1時刻サンプル branch を有効化可能）

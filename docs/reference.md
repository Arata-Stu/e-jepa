# vJEPA2 model + mask 統合コピー（src起点版, 2026-04-27）

プロジェクトrootから `src` を起点に分岐する構成に整理しています。

## 構成
- `src/models/` : ViT本体・predictor・model utils
- `src/masks/` : mask生成本体・mask適用
- `src/utils/` : modelが参照する最小utils（`tensors.py`）

## メモ
- `source/src/...` の二重構造は廃止
- model内部 import は `src.models...` を向くように調整済み

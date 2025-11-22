# Depth-Anything-3 利用方法まとめ (更新)

このドキュメントは、ローカル（WSL2 / Ubuntu22.04）環境で Depth-Anything-3（以降 DA3）を動かすために行った手順の記録と、実際の実行ログ、トラブル対応、次の作業計画をまとめた最新のマニュアルです。

**対象環境**
- ホスト: Windows 11
- GPU: NVIDIA GeForce RTX 1660 Ti（6GB）
- WSL2: Ubuntu 22.04（推奨実行環境）
- Python: conda 環境（推奨: Python 3.10）

---

## 1. 実施済みの主要作業（要約）

- GPU/ドライバ確認: `nvidia-smi`（Driver 528.79 / CUDA 12.0）を確認。
- 実行環境: WSL2 上の conda 環境で動作させる方針に決定。
- 依存関係対処: PyTorch の cu118 ビルドを推奨。`xformers` は依存衝突のため不要とした。
- 大量画像 OOM 対応: `process_batched.py` を作成し、画像を小バッチに分割して `da3 images` を順次実行する運用を採用。
- 可視化・検査ツール追加: `view_npz.py` を作成（後述の修正で複数ファイル・単一フレーム対応を適用）。

---

## 2. 追加・修正したスクリプト

- `scripts/sample_and_interleave.sh`
  - 2台のカメラディレクトリから指定サンプリング率でインターレーブされた入力フォルダを作るスクリプト。
  - 修正: `xy` サブフォルダを自動検出し、画像ファイルのみコピー。ディレクトリを誤って cp する不具合を解消。

- `process_batched.py`
  - 入力フォルダ内の画像を `--batch-size` ごとに分割して一時ディレクトリにコピーし、`da3 images` を呼び出す。実行後は一時ディレクトリを削除。
  - 設定例: `--batch-size 10`, `--process-res 336`, `--export-format mini_npz` が 6GB GPU では現実的な出発点。

- `view_npz.py`（更新）
  - 複数 `.npz` を受け取れるように `nargs='+'` を適用。
  - `depth` が `(N,H,W)` と `(H,W)` の両方に対応するように修正。
  - `--visualize --output-dir` で PNG を保存するワークフローを用意。

- `src/depth_anything_3/utils/export/__init__.py`
  - `colmap` エクスポートを遅延インポートに変更。`pycolmap` がない環境でも CLI の起動を妨げないようにした。

---

## 3. 本日の実行ログ（要約）

- 実行日: 2025-11-22
- テスト対象 name: `0914_02`（2カメラデータをインターレーブして入力）
- バッチ処理実行: `process_batched.py` を用いて全入力を処理
  - 実行結果: 343 バッチが順次完了
  - 出力ルート: `/mnt/c/data/da3_output/0914_02`
  - 合計 `.npz` ファイル: 343（各バッチに `results.npz`）

- サンプルファイルの中身（例: `batch_0000/exports/mini_npz/results.npz`）:
  - `depth`: `(10, 336, 336)` float32（min/max ≈ 0.30 / 2.31）
  - `conf`:  `(10, 336, 336)` float32
  - `extrinsics`: `(10, 3, 4)`
  - `intrinsics`: `(10, 3, 3)`

- 所見:
  - バッチ単位の npz はバッチに含まれるフレームをまとめて保持する形式。`view_npz.py` での可視化と最小限の統計確認では問題なし。

---

## 4. 実行手順（すぐ使えるコマンド例）

- 環境作成（WSL）
```bash
conda create -n da3 python=3.10 -y
conda activate da3
```

- PyTorch（cu118 推奨）
```bash
pip uninstall -y torch torchvision xformers
pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.3.1 torchvision==0.18.1
```

- リポジトリを開発モードでインストール
```bash
cd /mnt/c/work/Depth-Anything-3
pip install -e .
```

- インターレーブ（例: step=5）
```bash
cd /mnt/c/work/Depth-Anything-3/scripts
chmod +x sample_and_interleave.sh
./sample_and_interleave.sh /mnt/c/data/camera /mnt/c/data/da3_input 5
```

- バッチ処理（例）
```bash
cd /mnt/c/work/Depth-Anything-3
mkdir -p /mnt/c/data/da3_output/0914_02
python3 process_batched.py /mnt/c/data/da3_input/0914_02 \
  --output-dir /mnt/c/data/da3_output/0914_02 \
  --model-dir depth-anything/DA3-SMALL \
  --batch-size 10 --process-res 336 --export-format mini_npz --device cuda | tee /mnt/c/data/da3_output/0914_02/run_da3.log
```

- NPZ の中身確認（例）
```bash
python3 - <<'PY'
import numpy as np
f = '/mnt/c/data/da3_output/0914_02/batch_0000/exports/mini_npz/results.npz'
z = np.load(f, mmap_mode='r')
print('keys:', list(z.keys()))
for k in z.files:
    a = z[k]
    print(k, getattr(a,'shape',None), getattr(a,'dtype',None))
z.close()
PY
```

- サンプル可視化（少数ファイル）
```bash
mkdir -p /mnt/c/data/da3_output/0914_02/view_images
python3 view_npz.py /mnt/c/data/da3_output/0914_02/all_frames_test/*.npz --visualize --output-dir /mnt/c/data/da3_output/0914_02/view_images
ls -1 /mnt/c/data/da3_output/0914_02/view_images | sed -n '1,200p'
```

---

## 5. トラブルシューティング（要点）

- CUDA OOM 対処
  - `--batch-size` を下げる（例: 10→5→3）
  - `--process-res` を下げる（336→288→256）
  - `--export-format` を `mini_npz` のみにして中間重い処理を避ける

- Hugging Face モデルアクセスが 401 の場合
  - `--model-dir` に正しい repo id（例: `depth-anything/DA3-SMALL`）を使う
  - プライベート or gated モデルなら `HUGGINGFACE_HUB_TOKEN` を設定（`huggingface-cli login`）

- Windows ↔ WSL の I/O が遅い
  - 大量 I/O は WSL 側（例: `~/work`）に置いてから処理する方が安定

---

## 6. 次にやるべきこと（提案・優先度付き）

1. (A) **視覚確認**: `view_images` を確認して品質が問題なければ、
   - (A1) **全フレームを per-frame npz に展開**（必要なら float16 圧縮）
   - 実行コマンドは `all_frames` への一括展開スクリプトを用意済み（要実行）

2. (B) **本番処理の堅牢化**: `process_batched.py` に失敗バッチのリトライとログ出力強化を追加

3. (C) **3D 出力拡張**: GLB / gsplat 出力を必要に応じて導入（`pycolmap` / `gsplat` の導入手順を作成）

4. (D) **出力アーカイブ**: 完了したバッチ出力の整理とアーカイブ（ZIP/移動）

---

## 7. 変更履歴（簡易）

- 2025-11-22: バッチ処理スクリプト運用で `0914_02` を全件処理（343 バッチ成功）。
- 2025-11-22: `sample_and_interleave.sh`、`view_npz.py`、`export/__init__.py` を修正・改善。

---

この `DA3利用方法_UPDATED.md` を元に `DA3利用方法.md` を上書きするか、README に統合します。どちらを希望しますか？

また、次に進める作業（A/B/C/D）を選んでください。私がパッチ作成・実行スクリプトの実行・追加修正を続けます。

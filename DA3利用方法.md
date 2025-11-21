# Depth-Anything-3 利用方法まとめ

このドキュメントは、ローカル（WSL2 / Ubuntu22.04）環境で Depth-Anything-3（以下 DA3）を動かすために行った手順の記録と、今後の作業指針をまとめたものです。

**対象環境**
- ホスト: Windows 11
- GPU: NVIDIA GeForce RTX 1660 Ti（6GB）
- WSL2: Ubuntu 22.04
- Python: conda 環境（推奨: Python 3.10）

---

## 1. これまでにやったこと（実施済み）

1. GPU/ドライバ確認
   - `nvidia-smi` を Windows と WSL の両方で実行し、Driver Version: 528.79（CUDA 12.0）を確認。

2. 実行方針決定（WSL2上で実行）
   - WSL2(Ubuntu22) 上で conda 環境を作成し実行する方針に変更。
   - 理由: Linuxネイティブの方が依存解決と I/O 性能で安定。

3. PyTorch / xformers の依存整備
   - cu118（CUDA 11.8）ビルドの PyTorch を推奨 (例: `torch==2.3.1`, `torchvision==0.18.1`)。
   - xformers はバージョン依存で conflict が出たため、「xformers を入れない」選択を推奨（xformers がなくても動作する）。
   - 競合で問題が出た際には torch を 2.3.0 に下げて xformers を合わせる代替案も提示。

4. リポジトリインストール
   - WSL のローカルパスで `pip install -e .` を実行してパッケージを開発モードでインストール（必要に応じて `pip install -e "[app]"` を追加）。

5. CLI動作確認と入出力ハンドラ確認
   - `da3 auto` / `da3 images` / `da3 video` コマンドの使い方を確認。
   - `src/depth_anything_3/services/input_handlers.py` で画像・動画・COLMAPの入力処理を確認。

6. 大量画像処理でのOOM観測と対処
   - 8567 枚一括処理で CUDA OOM（プロセスがKilled）。原因: CLI は入力画像を全て一括で前処理し、巨大なテンソルを作る設計。
   - 対策案: 画像数を間引く、解像度を下げる、バッチ分割で順次処理する。

7. 小規模動作検証
   - 10 枚、50 枚などで試行。10 枚は成功（モデルは正常動作）。50 枚で OOM が出るケースあり→解像度やバッチサイズ調整が必要。

8. バッチ処理スクリプト追加
   - `process_batched.py` を作成：入力ディレクトリの画像を N 枚ずつのバッチに分けて順次 `da3 images` を実行する。各バッチ完了後に一時入力を削除。
   - これにより OOM を回避しつつ全枚数処理が可能になった。

9. NPZ確認ツール追加
   - `view_npz.py` を追加。`npz` の中身を一覧し、`depth` を画像化して出力できる。

---

## 2. 実行時によく使うコマンド（まとめ）

※WSL 上での実行を前提（Windows の C:\path は `/mnt/c/path` に置き換え）。

### conda 環境（例）
```bash
conda create -n da3 python=3.10 -y
conda activate da3
```

### PyTorch（cu118 推奨）
```bash
pip uninstall -y torch torchvision xformers
pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.3.1 torchvision==0.18.1
# xformers は入れないか、互換版が必要なら torch を下げて合わせる
```

### リポジトリをインストール
```bash
cd /mnt/c/work/Depth-Anything-3
pip install -e .
# Gradio を使う場合
pip install -e "[app]"
```

### 画像（ディレクトリ）を処理（小数枚テスト）
```bash
da3 images "/mnt/c/path/to/frames" \
  --image-extensions "jpg,jpeg" \
  --model-dir depth-anything/DA3-SMALL \
  --export-format mini_npz \
  --export-dir ./workspace/test_10 \
  --process-res 336 \
  --device cuda \
  --auto-cleanup
```

### 動画から処理（FPS指定）
```bash
da3 video "/mnt/c/path/to/video.mp4" \
  --fps 5 \
  --model-dir depth-anything/DA3-SMALL \
  --export-format glb-depth_vis \
  --export-dir ./workspace/car_run \
  --process-res 504 \
  --device cuda
```

### バッチ処理（先に作成したスクリプトを使う）
```bash
python3 process_batched.py \
  "/mnt/c/dataset/jetracer/0914_02_cam1/0914_02_cam1/xy" \
  --output-dir "./workspace/car_full" \
  --batch-size 10 \
  --process-res 336 \
  --export-format "mini_npz-depth_vis" \
  --device cuda
```

### NPZの中身を確認・可視化
```bash
# 中身確認のみ
python3 -c "import numpy as np; data=np.load('workspace/test_10/scene.npz'); print(data.files); print([ (k,data[k].shape) for k in data.files ])"

# 可視化スクリプト（保存）
pip install matplotlib
python3 view_npz.py workspace/test_10/scene.npz --visualize --output-dir ./depth_images
```

---

## 3. よくあるトラブルと対処法

- CUDA OOM（最も頻出）
  - 対処: `--process-res` を下げる（504→432→384→336→288）、`batch-size` を下げる（process_batched.py の引数）、`--export-format` を `mini_npz` のみにする。
  - GPUの利用状況確認: `nvidia-smi`

- xformers インストールで依存衝突
  - 対処: xformers を使わずに実行する（DA3 は PyTorch の SDPA でも動作）。必要なら torch をダウングレードして xformers の推奨バージョンに合わせる。

- /mnt/c での I/O が遅い
  - 対処: データや出力を WSL 側のファイルシステム（例: `~/work` や `/home/<user>/work`）にコピーして実行すると速度改善。

- gsplat（Gaussian Splatting）関連
  - `gsplat` は `gs_ply`/`gs_video` に必要。Windows/WSLでのインストール・動作はやや難しいため、3DGSが必須なら別途環境整備を検討。

---

## 4. これからやるべきこと（優先度順）

1. (必須) 全データのバッチ処理を開始
   - `process_batched.py` で `--batch-size 10`, `--process-res 336`, `--export-format mini_npz-depth_vis` を推奨。
   - 実行時間の目安: 8567 枚 ÷ 10 = 約857 バッチ。1バッチあたり数秒〜十数秒（GPU負荷、解像度に依存）。合計 1時間〜数時間。

2. (検証) 出力の簡易チェック
   - いくつかのバッチの `*.npz` を `view_npz.py` で可視化して、深度/信頼度が妥当か確認。

3. (必要なら) GLB 出力を行う
   - 3D ビジュアルが必要なら `--export-format glb` を追加。ただし GLB 出力はメモリ/時間コストが上がるため `--batch-size` を下げるか `--process-res` を控えめにする。

4. (将来的) Gaussian Splatting 出力の検討
   - 3DGS (`gs_ply`/`gs_video`) が必要なら `gsplat` を導入（`pip install git+https://github.com/nerfstudio-project/gsplat.git@<commit>`）。動作検証が必要。

5. (運用改善) バッチスクリプトのログ・再実行機能強化
   - 現在は成功/失敗を表示するのみ。必要なら失敗バッチのリトライや統合出力ディレクトリ整理スクリプトを作成。

6. (オプション) バックエンド常駐サービスの利用
   - 連続的に複数ジョブを実行するなら `da3 backend` を使いモデルをGPUに常駐させる。メモリに余裕がある場合に有効。

7. (保守) 出力のアーカイブと移動
   - 出力ファイル群は容量を食うため、処理後に不要な中間ファイルを圧縮または別ドライブへ移動する。

---

## 5. 便利なチェック/デバッグコマンド

- GPU使用状況
```bash
nvidia-smi
```

- 処理済みバッチ数確認（リアルタイム）
```bash
watch -n 5 'ls -d /mnt/c/work/Depth-Anything-3/workspace/car_full/batch_* 2>/dev/null | wc -l'
```

- 特定バッチの NPZ を簡易表示
```bash
python3 -c "import numpy as np; d=np.load('workspace/car_full/batch_0000/scene.npz'); print(d.files); print({k:d[k].shape for k in d.files})"
```

---

## 6. 補足メモ

- 作成したファイル:
  - `process_batched.py` — バッチ処理用スクリプト
  - `view_npz.py` — NPZ内容確認 + 深度画像可視化

- important: 処理は WSL の Linux 側ファイルシステムに置くと I/O と安定性が良い（例: `~/work/...`）。ただし既にWindows側のパス(`/mnt/c/...`)で問題なく動いている場合はそのままでも可。

---

必要ならこの `DA3利用方法.md` を追加修正して README に統合したり、`process_batched.py` にリトライ機能を付けたりします。どれを優先して進めましょうか？

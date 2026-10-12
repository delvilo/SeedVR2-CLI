# SeedVR2-cli · v2.7.0 — 獨立 CLI 影像與影片超解析度放大工具

> 🤖 **聲明**：本專案由 AI 輔助架構重構、工程實作與效能優化完成。

SeedVR2 是基於 Linux 命令列的影像與影片超解析度放大工具。推理管線採用單張 NVIDIA GPU 運行，並支援 CPU 平行計算進行卸載（Offload）與前處理，具備目錄批次處理、有限視訊流式傳輸、模型記憶體復用、VAE Tiling 分塊編解碼、DiT BlockSwap 區塊交換，以及長影片中斷續傳（--resume）。

[English](README.md) | [中文說明檔](README_zh.md)

---

## 核心改進與修復重點

1. **徹底解決原專案 OOM（記憶體不足）崩潰問題**：
   - 支援多階動態 VAE Tiling 分塊（最低可收縮至 128px）與因果 3D 卷積切片（Causal Conv3D Slicing）。
   - 事務性自動 OOM 恢復機制：推理期間若遭遇 VRAM 峰值不足，自動攔截異常、將 Chunk/Tile 尺寸減半、動態增加 BlockSwap 卸載比例並自動重試，不再中途崩潰中斷。
   - 可以在 **Google Colab 免費 T4 GPU (15GB VRAM)** 以及 **純 CPU** 環境下順暢穩定運行。
2. **啟動前 VRAM 預先檢查（Pre-flight Check）**：
   - 在下載大檔案模型權重或配置緩衝區前，先預先檢查目標 GPU 的剩餘 VRAM 容量。
   - 若 VRAM 容量不足以原生載入該模型且未配置 BlockSwap，會提前終止並提供具體改善建議（例如建議改用 Q4_K_M GGUF 或加上 `--blocks_to_swap` 參數），避免耗費時間下載數十 GB 權重後直接 OOM 報錯。
3. **全面移除 `torchvision` 與 `torchaudio` 依賴**：
   - `torchaudio` 全面自需求清單中清除（無任何引用）。
   - `torchvision` 影像變換改用 **原生純 PyTorch Tensor 運算**（`torch.nn.functional.interpolate` 與張量切片），具備最佳效能。
   - 視訊解碼升級為 Meta 官方 **`TorchCodec` (VideoDecoder)**，直接解碼為 PyTorch 張量，消除 OpenCV 逐幀讀取的 Python GIL 瓶頸與記憶體搬移開銷。

4. **長影片斷點續傳支援（`--resume`）**：
   - 專為 Google Colab 免費 T4 執行階段設計。每 33 幀自動保存獨立視訊分段檢查點與進度清冊。
   - 若 Colab 閒置斷線或遭時長限制中斷，下次執行相同指令加上 `--resume` 即可自動跳過已完成的 Chunk，並自動由原片預讀時序尾巴無縫平滑接續。
   - 全部 Chunk 完成後，透過 FFmpeg Concat Demuxer 在不到 1 秒內無損拼接為最終輸出，並完整保留原始音訊。
---

## 模型最低 VRAM 需求表

系統在執行前會根據以下閾值自動比對顯示卡可用顯存：

| 模型名稱 | 精度類型 | 權重大小 | 最少需要 VRAM (無 BlockSwap) | Colab T4 GPU (15GB) 支援狀況 |
| :--- | :--- | :--- | :--- | :--- |
| `seedvr2_ema_3b-Q4_K_M.gguf` | Q4_K_M 量化 | ~1.86 GB | **5.0 GB** | ✅ 原生直接運行（最快且省顯存） |
| `seedvr2_ema_3b-Q8_0.gguf` | Q8_0 量化 | ~3.4 GB | **7.0 GB** | ✅ 原生直接運行 |
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | FP8 | ~3.4 GB | **7.0 GB** | ✅ 原生直接運行 |
| `seedvr2_ema_3b_fp16.safetensors` | FP16 | ~6.8 GB | **9.5 GB** | ✅ 原生直接運行 |
| `seedvr2_ema_7b-Q4_K_M.gguf` | Q4_K_M 量化 | ~4.3 GB | **8.5 GB** | ✅ 原生直接運行 |
| `seedvr2_ema_7b-Q8_0.gguf` | Q8_0 量化 | ~7.8 GB | **12.0 GB** | ✅ 原生直接運行 |
| `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` | 混合 FP8/FP16 | ~7.5 GB | **11.5 GB** | ✅ 原生直接運行 |
| `seedvr2_ema_7b_fp16.safetensors` | FP16 | ~14.8 GB | **18.0 GB** | ⚠️ **需搭配 `--blocks_to_swap 32`**（未開會 OOM） |

---

## 在 Google Colab 上使用

開啟 [Colab 筆記本](colab/SeedVR2Lite_Colab.ipynb)，選擇免費的 **T4 GPU** 執行階段，依序執行儲存格即可。
- 支援 Google Drive 持久化存儲模型權重與生成結果。
- 內建 FFmpeg 9 與可選的 SageAttention / FlashAttention 檢查。

---

## CLI 命令列使用範例

在相容的 Linux / CUDA 環境中：

```bash
# 安裝相依套件
python -m pip install -r requirements.txt

# 查看所有說明參數
python inference_cli.py --help

# 1. 單張圖片放大 (例: 指定短邊 960px)
python inference_cli.py input.png --resolution 960 --output output.png

# 2. 影片高畫質放大至 1080p (搭配 FFmpeg 串流輸出)
python inference_cli.py input.mp4 \
  --resolution 1080 --batch_size 5 --chunk_size 33 \
  --temporal_overlap 3 --video_backend ffmpeg \
  --output output.mp4

# 3. 低顯存極限配置 (例如 8GB 顯卡或 Colab 跑 7B 模型)
python inference_cli.py input.mp4 \
  --dit_model seedvr2_ema_7b-Q4_K_M.gguf \
  --blocks_to_swap 32 --swap_io_components \
  --dit_offload_device cpu --vae_offload_device cpu \
  --resolution 1080

# 4. 目錄批次處理 (開啟模型跨檔復用)
python inference_cli.py media_folder/ \
  --output processed/ \
  --cache_dit --cache_vae \
  --dit_offload_device cpu --vae_offload_device cpu
```

---

## 常用參數說明

| 參數 | 說明 |
| :--- | :--- |
| `--dit_model` | 選擇 DiT 模型（3B 或 7B，可選 FP16、FP8 或 GGUF 量化版本）。預設為 3B FP8。 |
| `--resolution` | 設定放大後的短邊目標解析度（像素，例如 720、1080、1440）。 |
| `--batch_size` | 每次模型推理的訊框批次大小（必須符合 4n+1，如 1, 5, 9, 13, 17, 33）。數值越大時序連續性越好。 |
| `--chunk_size` | 單次加載的輸入最大訊框數（0 為自動記憶體規劃，最多 33 幀）。 |
| `--vae_encode_tiled` / `--vae_decode_tiled` | 啟用 VAE 分塊處理，大幅降低高解析度放大時的顯存消耗（預設啟用）。 |
| `--blocks_to_swap` | BlockSwap 卸載至系統 RAM 的變壓器層數量（3B 最多 32 層，7B 最多 36 層）。 |
| `--dit_offload_device cpu` | 當 DiT 空閒時將模型卸載至系統 RAM，釋放 VRAM 給 VAE 使用。 |
| `--vae_offload_device cpu` | 當 VAE 空閒時將模型卸載至系統 RAM，釋放 VRAM 給 DiT 使用。 |
| `--cache_dit` / `--cache_vae` | 在批次處理不同檔案時，將模型保留於記憶體中避免重複載入。 |
| `--color_correction` | 色彩校正演算法：`lab`（預設，色彩逼真度最高）、`wavelet`、`wavelet_adaptive`、`none`。 |
| `--video_backend` | 視訊編碼器後端：`opencv` 或 `ffmpeg`（支援 `--10bit` 高動態範圍）。 |
| `--debug` | 輸出詳細除錯訊息、時延分析與顯存即時監控數據。 |
| `--resume` | 開啟長影片中斷續傳功能。自動掃描並接續已完成的分段檢查點，全片完成後 1 秒無損合併。 |
| `--nvenc` | 開啟 NVIDIA GPU 硬體加速 H.265 編碼（hevc_nvenc），採用 P7 HQ 旗艦預設，輸出極速且接近視覺無損。 |

### 注意力機制與預編譯 FlashAttention-2 安裝指令

SeedVR2-cli 支援多種注意力後端（`--attention_mode`）：
- **`sageattn_2`**（預設）：速度極快，透過 `pip install sageattention` 即可秒裝。SM 8.0+（L4 / A100）原生全速支援，T4 (SM 7.5) 會自動平滑回退至 SDPA。
- **`sdpa`**：PyTorch 原生縮放點積注意力（內建，免安裝任何外部套件）。
- **`flash_attn_2`**：針對 Ampere、Ada、Hopper 架構（SM 8.0+）的高效 CUDA 注意力。**切勿從源碼編譯**（編譯需 25 分鐘以上）。請直接安裝對應環境的官方或社群預編譯 Wheel 輪子：

**Google Colab 環境（Ubuntu 24.04 / Python 3.13 / PyTorch 2.11 / L4 或 A100 GPU）：**
```bash
pip install --no-deps https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu128torch2.11-cp313-cp313-linux_x86_64.whl
```

**標準 Linux / CUDA 12.x / PyTorch 2.4+ 本地環境：**
```bash
pip install flash-attn --no-build-isolation
# 或至官方倉庫下載對應版本輪子：https://github.com/Dao-AILab/flash-attention/releases
```


### 模型顯存需求（啟動前檢查）

| 模型名稱 | 精度格式 | 權重體積 | 最低顯存（不開 BlockSwap） | Colab T4 (15 GB) | Colab L4 (22 GB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `seedvr2_ema_3b-Q4_K_M.gguf` | Q4_K_M | ~1.86 GB | **5.0 GB** | ✅ 原生直跑 | ✅ 原生直跑（**最推薦**） |
| `seedvr2_ema_3b-Q8_0.gguf` | Q8_0 | ~3.4 GB | **7.0 GB** | ✅ 原生直跑 | ✅ 原生直跑 |
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | FP8 | ~3.4 GB | **7.0 GB** | ✅ 原生直跑 | ✅ 原生直跑 |
| `seedvr2_ema_3b_fp16.safetensors` | FP16 | ~6.8 GB | **9.5 GB** | ✅ 原生直跑 | ✅ 原生直跑 |
| `seedvr2_ema_7b-Q4_K_M.gguf` | Q4_K_M | ~4.3 GB | **8.5 GB** | ✅ 原生直跑 | ✅ 原生直跑 |
| `seedvr2_ema_7b-Q8_0.gguf` | Q8_0 | ~7.8 GB | **12.0 GB** | ✅ 原生直跑 | ✅ 原生直跑 |
| `seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors` | Mixed FP8/FP16 | ~7.5 GB | **11.5 GB** | ✅ 原生直跑 | ✅ 原生直跑 |
| `seedvr2_ema_7b_fp16.safetensors` | FP16 | ~14.8 GB | **18.0 GB** | ⚠️ 需 `--blocks_to_swap 32` | ✅ 原生直跑 |

### L4 GPU 1080p 影片放大效能基準與實測（NVIDIA L4 · 22 GB VRAM · SM 8.9 Ada Lovelace）

於 Google Colab L4 環境實測，基準任務為將 640×360 視訊放大至 **1080p (1920×1080)**。L4 預設使用硬體原生的 SageAttention 2 注意力機制。

| 測試配置 | 模型檔案 | 批次大小 (Batch) | `--keep_on_vram` | 顯存峰值 (Peak) | DiT 推理耗時 | 全流程總耗時 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **3B Q4 卸載基準** | `seedvr2_ema_3b-Q4_K_M.gguf` | 5 | ❌ (CPU 卸載) | 5.01 GB | 7.95 秒 | 62.88 秒 |
| **3B Q4 顯存常駐** | `seedvr2_ema_3b-Q4_K_M.gguf` | 5 | ✅ | 6.94 GB | 6.59 秒 | **55.84 秒** |
| **3B Q4 批次 9** | `seedvr2_ema_3b-Q4_K_M.gguf` | 9 | ✅ | 8.39 GB | 9.05 秒 (1.0秒/幀) | 89.20 秒 |
| **3B Q4 批次 13** | `seedvr2_ema_3b-Q4_K_M.gguf` | 13 | ✅ | 8.46 GB | 11.48 秒 (**0.88秒/幀**) | 122.86 秒 |
| **7B Q4 平衡推薦** | `seedvr2_ema_7b-Q4_K_M.gguf` | 5 | ✅ | 9.49 GB | 9.94 秒 | 72.66 秒 |
| **7B Q4 批次 9** | `seedvr2_ema_7b-Q4_K_M.gguf` | 9 | ✅ | 10.95 GB | 13.61 秒 (1.5秒/幀) | 95.80 秒 |
| **7B Q8 高畫質** | `seedvr2_ema_7b-Q8_0.gguf` | 5 | ✅ | 13.29 GB | 9.60 秒 | 84.79 秒 |
| **7B Q8 批次 9** | `seedvr2_ema_7b-Q8_0.gguf` | 9 | ✅ | 14.75 GB | 13.20 秒 (1.46秒/幀) | 96.11 秒 |
| **7B FP16 卸載模式** | `seedvr2_ema_7b_fp16.safetensors` | 5 | ❌ (CPU 卸載) | 13.42 GB | 29.37 秒 | 101.83 秒 |
| **7B FP16 顯存常駐** | `seedvr2_ema_7b_fp16.safetensors` | 5 | ✅ | 20.41 GB | 8.90 秒 | **63.67 秒** |

#### 大模型專題實測（7B Q8_0 與 7B FP16 在 L4 GPU）
- **`seedvr2_ema_7b-Q8_0.gguf` (~8.3 GB)**：
  - **完全可行**！開啟 `--keep_on_vram` 且批次可開至 `--batch_size 9`，顯存峰值僅 14.75 GB，距離 22 GB 顯存上限有超過 7 GB 的安全餘量。畫質與 FP16 無肉眼差異，為大模型最高性價比首選。
- **`seedvr2_ema_7b_fp16.safetensors` (~15.8 GB)**：
  - **在 `--batch_size 5` 下可行**！常駐顯存峰值達 20.41 GB，全流程 63.67 秒。
  - **在 `--batch_size 9` 下不可行**！預檢系統會精準偵測需 ~22.6 GB 顯存並主動安全中止（提供調整建議，不自動盲目 fallback）。
  - **建議運作參數**：欲常駐運行 7B FP16 請務必保持 `--batch_size 5` 或更小；若需更大批次處理長影片，請關閉 `--keep_on_vram` 或改用 `7b-Q8_0`。

---

#### L4 GPU 四大標準最佳參數推薦

##### 1. 最佳畫質 (Best Quality) — 7B Q8-0, Batch 9, 1080p
```bash
python inference_cli.py input.mp4 \
  --dit_model seedvr2_ema_7b-Q8_0.gguf \
  --resolution 1080 --batch_size 9 \
  --attention_mode sageattn_2 --keep_on_vram \
  --video_backend ffmpeg --output output_best_quality.mp4
```

##### 2. 最快時間 (Fastest Time / 最大吞吐量) — 3B Q4, Batch 13, 1080p
```bash
python inference_cli.py input.mp4 \
  --dit_model seedvr2_ema_3b-Q4_K_M.gguf \
  --resolution 1080 --batch_size 13 \
  --attention_mode sageattn_2 --keep_on_vram \
  --video_backend ffmpeg --output output_fastest.mp4
```

##### 3. 長影音 (Long Video / 多場景多分鐘長片) — 7B Q4, Chunk 33, Overlap 3
```bash
python inference_cli.py input.mp4 \
  --dit_model seedvr2_ema_7b-Q4_K_M.gguf \
  --resolution 1080 --batch_size 9 \
  --chunk_size 33 --temporal_overlap 3 \
  --attention_mode sageattn_2 --keep_on_vram \
  --video_backend ffmpeg --output output_long_video.mp4
```

##### 4. 短影音 (Short Video / 數秒鏡頭極限修復) — 7B FP16, Batch 5
```bash
python inference_cli.py input.mp4 \
  --dit_model seedvr2_ema_7b_fp16.safetensors \
  --resolution 1080 --batch_size 5 \
  --attention_mode sageattn_2 --keep_on_vram \
  --video_backend ffmpeg --output output_short_video.mp4
```
---

## 專案歷史與致謝

本專案衍生自 [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) 與 [ByteDance SeedVR](https://github.com/ByteDance-Seed/SeedVR)。遵循 [LICENSE](LICENSE) 所列之 Apache 2.0 開源協議。

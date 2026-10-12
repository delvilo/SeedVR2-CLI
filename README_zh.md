# SeedVR2-cli — 獨立 CLI 影像與影片超解析度放大工具

SeedVR2 是基於 Linux 命令列的影像與影片超解析度放大工具。推理管線採用單張 NVIDIA GPU 運行，並支援 CPU 平行計算進行卸載（Offload）與前處理，具備目錄批次處理、有限視訊流式傳輸、模型記憶體復用、VAE Tiling 分塊編解碼、DiT BlockSwap 區塊交換，以及可選的 torch.compile 加速。

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
   - `torchvision` 影像變換改用 **原生純 PyTorch Tensor 運算**（`torch.nn.functional.interpolate` 與張量切片），具備最佳效能且高度相容於 `torch.compile`。
   - 視訊解碼升級為 Meta 官方 **`TorchCodec` (VideoDecoder)**，直接解碼為 PyTorch 張量，消除 OpenCV 逐幀讀取的 Python GIL 瓶頸與記憶體搬移開銷。

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

### L4 GPU 效能基準與實測結果（NVIDIA L4 · 22 GB VRAM · SM 8.9 Ada Lovelace）

於 Google Colab L4 環境實測。輸入影像：320×478 px，固定 `--seed 42`，啟用 VAE 分塊與 CPU 卸載。扣除初次下載權重耗時。L4（SM 8.9）預設使用 SageAttention 2 注意力機制。

| 模型與量化 | 注意力機制 | 輸出解析度 | 總流程耗時 | DiT 推理階段 |
| :--- | :--- | :---: | :---: | :---: |
| 3B Q4_K_M | SDPA | 360p (538px) | ~28 秒 | 3.3 秒 |
| **3B Q4_K_M** | **SageAttn 2** | **360p** | **~16.5 秒** ✅ | **6.8 秒** |
| **3B Q4_K_M** | **SageAttn 2** | **720p** | **~16.3 秒** ✅ | **5.1 秒** |
| 3B Q8_0 | SageAttn 2 | 360p | ~17 秒 | 4.1 秒 |
| 3B Q8_0 | SageAttn 2 | 720p | ~16.6 秒 | 5.4 秒 |
| 7B Q4_K_M | SageAttn 2 | 360p | ~19 秒 | 6.8 秒 |
| 7B Q4_K_M | SageAttn 2 | 720p | ~18.4 秒 | 7.1 秒 |
| 7B FP8 mixed | SageAttn 2 | 360p | ~19 秒 | 5.6 秒 |
| 7B FP8 mixed | SageAttn 2 | 720p | ~18.4 秒 | 6.9 秒 |

**L4 實測核心結論：**
1. **SageAttention 2 效能提升顯著**：相較於 SDPA，SageAttention 2 在 L4 上將完整單圖端到端時間由 28 秒降至 16.5 秒（加速 **1.7 倍**）。
2. **720p 與 360p 耗時近乎無差**：因 L4 配備 22GB 顯存，720p 運算未觸發頻繁的分塊換頁，耗時維持在 ~16.3 秒。
3. **3B Q4_K_M 為速度最優選**：端到端僅需約 16 秒，顯存佔用低，最適合批量處理與長影片放大。
4. **7B 模型在 L4 表現優異**：7B Q4_K_M 與 7B FP8 放大 720p 僅需 18.4 秒，只比 3B 多約 2 秒，但畫面細節豐富度大幅提高。
5. **7B FP8 vs 7B Q4**：兩者在 L4 上速度幾乎相同（~18.4 秒），建議優先採用 Q4_K_M。

**L4 推薦執行命令（極速平衡：3B Q4，720p）：**
```bash
python inference_cli.py input.png \
  --dit_model seedvr2_ema_3b-Q4_K_M.gguf \
  --resolution 720 --attention_mode sageattn_2 \
  --model_dir /content/models --output output.png
```

**L4 高畫質推薦命令（畫質優先：7B Q4，720p）：**
```bash
python inference_cli.py input.png \
  --dit_model seedvr2_ema_7b-Q4_K_M.gguf \
  --resolution 720 --attention_mode sageattn_2 \
  --model_dir /content/models --output output.png
```
---

## 專案歷史與致謝

本專案衍生自 [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) 與 [ByteDance SeedVR](https://github.com/ByteDance-Seed/SeedVR)。遵循 [LICENSE](LICENSE) 所列之 Apache 2.0 開源協議。

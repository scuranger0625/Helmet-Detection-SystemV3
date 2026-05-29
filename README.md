# Helmet-Detection-SystemV3

# BOS 驅動之自適應即時推論系統
## BOS-Driven Adaptive Inference Scheduling System for Real-Time Helmet Violation Detection

Helmet-Detection-SystemV3 已不再只是傳統的 frame-by-frame YOLO 偵測專案。

本專案的核心目標，是將即時視覺推論問題重新建模為：

- 線上排程問題（Online Scheduling Problem）
- 自適應推論問題（Adaptive Inference Problem）
- CPU 平行推論系統（CPU Parallel Inference System）

而不是單純追求：

```text
每一幀都硬跑 YOLO
```

---

# 專案核心概念

本專案將 YOLO detector 視為：

```text
昂貴的瓶頸資源（Bottleneck Resource）
```

因此系統的核心不再是：

```text
image -> YOLO -> output
```

而是：

```text
Video Stream
    ↓
Cheap Feature Layer
(motion / ROI / entropy / count)
    ↓
SystemState Builder
    ↓
BOS Scheduler
    ↓
Priority Queue / Dispatch
    ↓
Worker Pools
(detector / classifier / I/O)
    ↓
Cache & Result Layer
```

Scheduler 將動態決定：

- 哪些 frame 值得花 detector 成本
- 哪些 frame 可以直接沿用 cache
- 哪些 job 可以延後
- 哪些狀態需要強制 refresh
- 何時系統已接近 budget 上限

---

# 研究定位

本專案並不只是：

```text
YOLO + Motion Threshold
```

而是：

```text
BOS-Driven CPU Parallel Inference Scheduling System
```

核心研究方向包含：

- Real-Time Computer Vision
- Online Scheduling
- Adaptive Inference
- Event-Driven Scheduling
- Queue-Aware Scheduling
- CPU Parallel Inference
- Cache-Aware Dispatch
- Detector Budget Control

---

# 主要功能

# 1. Adaptive Inference Scheduling

V3 不再使用：

```python
if motion > threshold:
    detect()
```

而是使用：

```python
observe(SystemState)
    -> enqueue
    -> reuse_cache
    -> force_refresh
    -> drop
```

這代表系統已從：

- heuristic trigger script

進化成：

- stateful scheduling system

---

# 2. Detection Cache

當場景變化不大時，系統會沿用近期 detection cache。

此設計可降低：

- detector calls
- CPU/GPU 推論成本
- 冗餘 YOLO inference

並提升整體 FPS 與即時性。

---

# 3. BOS-Inspired Urgency Scheduling

系統會根據以下資訊計算 urgency score：

- motion_score
- cache_age
- count_change
- classifier_entropy
- tracker_confidence
- budget_pressure
- ROI changes

範例：

```text
U_t =
wm * motion
+ wa * cache_age
+ wc * count_change
+ wu * entropy
+ wr * risk
- wb * budget_pressure
```

當 urgency 達到門檻時，才會觸發完整 detector 推論。

---

# 4. 人車配對（Human-Motorcycle Pairing）

系統會：

1. 偵測 person + motorcycle
2. 執行 rider pairing
3. 裁切 head crop
4. 估計安全帽違規風險

---

# 5. Head Risk Estimation

目前 V3 使用 heuristic-based risk estimator：

- skin ratio
- edge density
- texture variance
- brightness statistics
- reflection analysis

未來版本預計導入：

- MobileNetV3
- EfficientNet
- ONNX Runtime
- OpenVINO
- Quantized inference

---

# 專案結構

```text
Helmet-Detection-SystemV3/
│
├── api/
├── core/
├── runtime/
├── src/
├── templates/
├── utils/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 實驗指標

本專案重視的不只是 accuracy，而是整體系統效能：

- Average FPS
- Detect Ratio
- Detector Calls / Min
- CPU-ms / Min
- P95 / P99 Latency
- Frame-Level Recall
- Event-Level Recall

---

# 研究動機

大部分即時 AI 系統都存在大量：

```text
冗餘推論（Redundant Inference）
```

相鄰 frames 往往高度相似，但傳統系統仍然每幀執行完整 detector。

因此本研究希望探索：

- budget-aware inference
- adaptive detector dispatch
- queue-aware scheduling
- cache-aware inference reuse
- event-driven scheduling

如何降低推論成本，同時維持 acceptable recall。

---

# 未來工作

未來版本預計加入：

- MobileNetV3 Helmet Classifier
- ONNX Runtime Optimization
- OpenVINO Deployment
- Multi-worker Scheduling
- Priority Queue Dispatch
- Contextual Bandit Scheduling
- Queue-Aware Inference Control
- Edge Deployment Optimization

---

# 使用技術

- Python
- YOLOv8
- OpenCV
- Flask
- React / Vite
- NumPy
- Scheduling Heuristics
- Real-Time Video Processing

---

# 作者

Chen-Hong  
National Chung Cheng University  
Graduate Institute of Telecommunications and Communication

---

# 專案核心思想

本專案真正想研究的問題是：

```text
在有限 detector budget 下，
哪些 frame 值得花昂貴推論成本？
```

因此，V3 的核心不是 detector 本身，

而是：

```text
如何進行 BOS 驅動的即時推論排程。
```


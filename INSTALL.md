# AutoLabel Dock · 环境配置指南

> 这份文档面向"克隆代码后第一次配置环境"的场景。复制粘贴执行即可。
> 阅读对象：源码运行 / 二次开发。
> 只想使用单 exe 绿色版？直接跳到 [§5 发行包（绿色版）](#5-发行包绿色版)。

---

## 目录

1. [前置条件](#1-前置条件)
2. [完整依赖说明](#2-完整依赖说明)
3. [源码运行 · 三步走](#3-源码运行--三步走)
4. [可选：LocateAnything 文本标注后端](#4-可选locateanything-文本标注后端)
5. [发行包（绿色版）](#5-发行包绿色版)
6. [常见问题](#6-常见问题)

---

## 1. 前置条件

| 项目 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Windows 10/11、macOS 12+、Linux（Ubuntu 20.04+ 推荐） | 跨平台，PyQt5 本身支持三平台 |
| Python | **≥ 3.10**（推荐 3.10–3.12） | `python --version` 查看 |
| 磁盘 | 源码运行预留 ≥ 3 GB；模型 + LocateAnything 视使用场景另算 | 见各步骤 |
| GPU（强烈推荐） | NVIDIA 显卡 + 驱动；显存 ≥ 4 GB（训练 ≥ 6 GB） | 纯 CPU 也能跑，但训练与 LocateAnything 会很慢或不可用 |
| 显存（LocateAnything） | ≥ 6 GB（启用时空闲 ≥ 5 GB） | 见 §4 |

> **Windows 用户**：训练数据集准备会创建符号链接。建议开启"开发者模式"
> （设置 → 隐私和安全 → 开发者选项），可零拷贝；不开也能跑，代码会自动降级到硬链接或复制。

---

## 2. 完整依赖说明

仓库根目录的 `requirements.txt` 已经涵盖所有运行期依赖。安装之前请先理解每个包的角色：

### 2.1 基础依赖（必装）

| 包 | 版本 | 用途 | 备注 |
|---|---|---|---|
| `PyQt5` | ≥ 5.15 | GUI 框架（主窗口、画布、对话框） | GPL-3.0；商用需 Riverbank 商业授权 |
| `pyqtgraph` | ≥ 0.13 | 高速绘图（训练 loss/mAP 曲线） | 基于 Qt，与 PyQt5 配合 |
| `ultralytics` | ≥ 8.0 | YOLOv8 模型加载 / 推理 / 训练 | AGPL-3.0；首次使用自动下载 `yolov8n.pt` 等底模 |
| `torch` | 跟随 CUDA | 深度学习后端；ultralytics 间接依赖 | **必须**与本机 CUDA 匹配，详见 §3.1 |
| `numpy` | 任意 | 标注 / 张量运算 | ultralytics 已强制依赖 |
| `Pillow` | ≥ 9.0 | 图片读取、缩略图、用户脚本 API | 标注 JSON 不带图，纯图片 IO 用 Pillow |
| `opencv-python` | ≥ 4.5 | 视频抽帧、图像变换、轮廓检测 | 仅在视频导入与脚本工具面板中使用 |
| `PyYAML` | ≥ 6.0 | 解析 `data.yaml`（YOLO 数据集配置） | 训练时由 ultralytics 加载 |
| `packaging` | ≥ 20.0 | 版本比较（ultralytics 的版本探测） | ultralytics 已间接依赖 |

### 2.2 可选依赖（仅文本提示标注需要）

| 包 | 用途 |
|---|---|
| `transformers` (≥4.57, <4.58) | 加载 LocateAnything-3B（NVIDIA） |
| `accelerate` | Hugging Face 模型加载加速 |
| `bitsandbytes` | 4-bit 量化推理，节省显存 |
| `decord` | 高性能视频解码（LocateAnything 内部） |

这些包体积大、依赖重，**默认不安装**。详见 [§4](#4-可选locateanything-文本标注后端)。

### 2.3 开发依赖（只在改代码时需要）

| 包 | 用途 | 安装方式 |
|---|---|---|
| `pyinstaller` | 打包成 `.exe` / 可执行目录 | `pip install pyinstaller` |
| `pytest` | 跑测试 | `pip install pytest` |
| `mypy` / `ruff` | 类型检查 / lint | `pip install mypy ruff` |

---

## 3. 源码运行 · 三步走

### 3.0 克隆代码

```bash
git clone https://github.com/Sen-yin/autolabel-dock.git
cd autolabel-dock
```

### 3.1 创建虚拟环境并安装 PyTorch

> ⚠️ **PyTorch 必须先于 ultralytics 单独安装**，否则 pip 会装 CPU-only 版本。
> 到 <https://pytorch.org/get-started/locally/> 选好对应 CUDA 版本，复制官方命令。

**示例 · CUDA 12.1 + 纯 GPU：**

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**示例 · CUDA 11.8：**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**示例 · 仅 CPU（不推荐，训练极慢）：**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> 验证：`python -c "import torch; print(torch.cuda.is_available())"` 应输出 `True`（有 GPU 时）。

### 3.2 安装项目依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 会一次性装好 PyQt5、pyqtgraph、ultralytics、numpy、Pillow、opencv-python、PyYAML、packaging —— 这些都是项目运行所需的全部 Python 包。

> 💡 启用 LocateAnything 时再单独跑：
> `pip install -e ".[locateanything]"`
> 详见 §4。

### 3.3 启动应用

```bash
python main.py
```

首次启动会：

- 在用户主目录创建 `~/.autolabel/`，存放全局配置与日志
- Ultralytics 首次使用底模（如 `yolov8n.pt`）时自动下载到当前目录

> 离线环境请提前把底模放到仓库根目录或 `~/.config/Ultralytics/`。

---

## 4. 可选：LocateAnything 文本标注后端

LocateAnything-3B（NVIDIA）是开放词汇检测模型，可以用"红色安全帽"这样的自然语言预标注。**不启用它，应用其余功能完全正常。**

启用需同时满足以下三个条件：

### 4.1 安装可选 Python 依赖

```bash
pip install -e ".[locateanything]"
```

会额外装 `transformers`、`accelerate`、`bitsandbytes`、`decord`。

### 4.2 提前下载模型权重（必须）

应用以离线模式加载（`HF_HUB_OFFLINE=1`），**不会自动下载**。

```bash
# 需要先装 huggingface_hub（hf 是新版 CLI）
pip install -U "huggingface_hub[cli]"
hf download nvidia/LocateAnything-3B
# 旧版：huggingface-cli download nvidia/LocateAnything-3B
```

默认缓存位置：`~/.cache/huggingface/hub/`
也可用 `$HF_HOME` 或 `$HUGGINGFACE_HUB_CACHE` 自定义。

### 4.3 硬件要求

- NVIDIA GPU，`nvidia-smi` 必须可用
- **不支持 CPU**
- 总显存 ≥ 6 GB，启用时空闲显存 ≥ 5 GB

> LocateAnything 与 YOLO 模型不会同时占用 GPU —— 启用 LA 会自动卸载已加载的 YOLO 模型，反之亦然。它运行在独立子进程，与主界面隔离。

---

## 5. 发行包（绿色版）

`build.bat` 一键打包 `onedir` 模式到 `dist\AutoLabelDock\AutoLabelDock.exe`：

```bat
:: 在已激活的 venv 里
pip install pyinstaller
build.bat
```

输出：

```
dist\AutoLabelDock\AutoLabelDock.exe       :: 启动器（图标已嵌入 resources/icons/app.ico）
dist\AutoLabelDock\_internal\...           :: 依赖、资源等
```

分发时把整个 `dist\AutoLabelDock\` 文件夹压成 zip 给最终用户，不需要用户装 Python。

> - 改了 `resources/icons/app_icon.png` 想重新生成 `.ico`： `python tools\build_icon.py`；`build.bat` 会在缺失 `.ico` 时自动调用。
> - LocateAnything 后端的 torch/transformers 不进打包；目标机器启用 LA 前先 `pip install -e .[locateanything]`。

---

## 6. 常见问题

### Q: 启动报 `WinError 1114 DLL initialization failed — c10.dll`

A: PyQt5 把 Qt5/bin 提前插到 PATH，挤掉了 torch 的依赖路径。这是 Windows + Python 3.13 + torch 2.9 的已知组合问题。

解决：
- 升级 Python 到 3.12 或降级 torch 到 2.8
- 或直接用 Python 3.10/3.11（项目验证过的版本）

打包版的运行时已经在 `packaging/pyi_rth_dllpath.py` 做了预加载修复，源码运行版本不需要。

### Q: 训练时报 `Cannot find opener` 或 `opencv_videoio_ffmpeg500_64.dll not found`

A: OpenCV 自带的 FFmpeg DLL 没装上。重装：

```bash
pip install --force-reinstall opencv-python
```

### Q: `pip install -r requirements.txt` 报 ultralytics 与 torch 不兼容

A: 通常是 torch 装晚了。先按 §3.1 单独装 torch，再装其余依赖；或一次性：

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Q: 训练按钮灰着点不动

A: 检查三件事：

1. `python -c "import torch; print(torch.cuda.is_available())"` 是否为 `True`
2. 项目是否已有 ≥ 1 张已"确认"标注的图片（黄色虚线不算确认）
3. 类别是否已配置（项目设置 → 类别管理）

### Q: 想换源 / 用国内镜像

```bash
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -r requirements.txt
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 📋 一页纸速查

```bash
# === 源码运行（GPU 用户） ===
git clone https://github.com/Sen-yin/autolabel-dock.git
cd autolabel-dock
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python main.py

# === 可选：文本提示标注 ===
pip install -e ".[locateanything]"
hf download nvidia/LocateAnything-3B

# === 打包成 exe ===
pip install pyinstaller
build.bat
```

按这三段顺序执行，本机即可从源码跑通整个流程。

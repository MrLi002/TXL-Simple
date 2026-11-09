# TimeLens-XL-Large 精简版

按 OF-AF 的工程组织方式，单独提取 TimeLens-XL 的 `models/Expv8_large`。
这是工程精简，不是把 TimeLens-XL 的光流模块换成 OF-AF。

保留原 Large 网络的可训练层、参数名称、计算顺序、事件分组方式、左右
递推解码器、MaskNet 的两级光流修正和最终图像修复。参数量为 **7,871,125**。

## 1. 工程结构

| 文件 | 用途 |
|---|---|
| `models/model.py` | 唯一模型 `MyModel / TimeLensXLLarge`，直接接收两张图像和事件体素 |
| `models/encoder.py` | Large 的事件编码器、图像编码器 |
| `models/flow_decoder.py` | `FlowDecoder`、`IFBlock`、`MaskNet` |
| `models/blocks.py` | Large 使用的两种残差块 |
| `models/warp.py` | backward warp，保留上游采样公式 |
| `datasets/bsergb.py` | BS-ERGB 入口 |
| `datasets/gopro.py` | GOPRO 预计算体素入口 |
| `datasets/common.py` | 序列索引、裁剪、打包 |
| `util/events.py` | 必要的事件读取和体素组织 |
| `util/checkpoint.py` | 官方权重前缀转换、严格加载、续训状态 |
| `train_model.py` | 直接训练/微调/续训入口 |
| `bsergb_eval.py`、`evaluate.py` | 测试、保存预测与指标 |
| `interpolate.py` | 两张图像加已准备体素的单样本推理 |
| `tests/` | 数值回归及必要功能检查 |

删除了其他模型、注册器、`BaseModel`、大批实验参数文件、主机路径配置、
测速器、未调用的 warp 分支、历史注释代码、缓存文件、网页和演示资源。
没有引入 OF-AF 的 BiFEB、Query、BiW、forward splatting 或相关性体。

上游对应关系：

| 上游 `models/Expv8_large/archs/` | 精简后 |
|---|---|
| `XXNet_final_attenfusion_arch.py` | `models/model.py` |
| `encoder_module.py` | `models/encoder.py` |
| `flow_decoder.py` | `models/flow_decoder.py`、`models/blocks.py`、`models/warp.py` |
| `runExpv8_large.py`、外部框架 | 直接训练/测试脚本 |

## 2. 安装

建议 Python 3.10 或更高版本，先安装与你的 CUDA 对应的 PyTorch 和
torchvision，再安装：

```bash
pip install -r requirements.txt
```

无需 CuPy、自定义 CUDA 算子、相关性扩展或 numba。
LPIPS 沿用上游的 `pyiqa.create_metric('lpips')`，首次使用可能下载权重。
`--lpips_weight 0` 可关闭训练中的 LPIPS；测试默认只算 PSNR/SSIM。

## 3. 两个倍率必须分清

- **数据倍率**：两个端点之间有多少个真实帧间隔。`skip=3` 对应数据倍率 4，
  有 3 张 GT。训练用 `--data_ratios 4` 指定。
- **内部倍率**：模型的事件分组和递推网格，使用 `--interp_ratio` 指定。

例如官方 BS-ERGB x4 配置使用数据倍率 4、内部倍率 16。此时读入的事件
仍为 128 通道：每个真实帧间隔先编码为 32 通道，网络再组织为 16 组、
每组 8 通道；只输出内部节点 4、8、12 对应的三张图像。

`--interp_ratio 0` 表示内部倍率跟随样本的数据倍率，适合 GOPRO 混合倍率训练。
固定内部倍率必须是数据倍率的整数倍，且倍率必须整除 `voxel_bins`。
模型可接收浮点时间，但必须位于内部网格上；没有增加任意时刻光流插值。

## 4. 数据格式

根目录下每个场景包含 `images/` 和 `events/`。根目录也可以直接是单个场景。

| 相对路径示例 | 内容 |
|---|---|
| `scene_01/images/000000.png` | 第 0 张 RGB 图像 |
| `scene_01/images/000001.png` | 第 1 张 RGB 图像 |
| `scene_01/events/000000.npz` | 图像 0 到图像 1 之间的事件/体素 |

按自然文件名顺序配对第 i 个事件文件与第 i→i+1 张图像。文件缺失会破坏
这种对应，运行前应确认序列完整。目录名可用 `--image_dir` 和 `--event_dir` 修改。

### BS-ERGB

事件文件包含一维 `x / y / timestamp / polarity` 数组，时间升序，极性为 0/1。
默认 `--coordinate_mode legacy` 沿用 TimeLens-XL 对 BS-ERGB 固定点坐标的转换。
已经转换为像素坐标的导出数据，使用 `--coordinate_mode pixel`。

保留上游按每个真实帧间隔分箱、空间双线性累积及已知损坏区间排除规则。
没有改成 OF-AF 的整段事件体素化方式，也没有固定 5ms 分箱。

### GOPRO

使用 **TimeLens-XL 原有预计算格式**：每个真实帧间隔的 NPZ 包含 `data`，
形状为 `[C,H,W]` 或 `[H,W]`。每个区间根据目标通道数求和压缩或补零后拼接。

**不能直接将 OF-AF 的 `events_clean_8x_v2` 原始事件 NPZ 当作这个格式使用。**
本项目没有额外引入 v2e、视频超采样或数据集生成工程；已有官方预计算数据即可训练。

## 5. 训练与微调

GOPRO 固定 x16：

```bash
python train_model.py --dataset gopro --data_root /path/to/GOPRO_TRAIN \
  --data_ratios 16 --interp_ratio 16 --epoch 20 --lr 1e-4 \
  --batch_size 1 --crop_size 256 --save_dir train/gopro_large
```

沿用上游 2/4/8/16/32 混合倍率与采样概率：

```bash
python train_model.py --dataset gopro --data_root /path/to/GOPRO_TRAIN \
  --data_ratios 2 4 8 16 32 --interp_ratio 0 --epoch 20 --lr 1e-4 \
  --batch_size 1 --save_dir train/gopro_large_mix
```

从 GOPRO 权重在 BS-ERGB 上微调：

```bash
python train_model.py --dataset bsergb --data_root /path/to/BSERGB_TRAIN \
  --data_ratios 4 --interp_ratio 16 --epoch 20 --lr 1e-5 \
  --crop_size 384 --batch_size 1 \
  --weights_path train/gopro_large/epoch19.pth --only_weights \
  --save_dir train/bsergb_large
```

也可以将 `--weights_path` 换成官方 `Expv8_large` 的 `.pt` 文件。
`--weights_path` 始终只读取模型权重；不会沿用旧优化器和学习率计划。
学习率示例是可调实验配置，不代表官方最优超参数。

默认损失为 Charbonnier + 0.1 LPIPS，梯度裁剪阈值 0.01，默认不启用 AMP。
`--amp` 是可选运行模式；CPU 调试可用 `--device cpu --num_worker 0 --lpips_weight 0`。
训练保存 `epochN.pth`、`last.pth`、`args.json` 和 `history.json`。

### 断点续训

```bash
python train_model.py --dataset bsergb --data_root /path/to/BSERGB_TRAIN \
  --data_ratios 4 --interp_ratio 16 --epoch 20 --lr 1e-5 \
  --crop_size 384 --batch_size 1 \
  --resume train/bsergb_large/last.pth --save_dir train/bsergb_large
```

续训恢复模型、优化器、scheduler、AMP scaler 和随机状态，从已完成 epoch 的
下一轮开始。`--epoch` 是原训练计划的总轮数，续训时保持不变。
若改变总轮数、数据倍率、裁剪等计划，使用 `--weights_path` 开始新实验。
恢复以已保存的完整 epoch 为边界，不恢复未保存 epoch 内的数据迭代位置。

## 6. 测试与保存预测

```bash
python bsergb_eval.py --checkpoint train/bsergb_large/epoch19.pth \
  --test_root /path/to/BSERGB_TEST --interp_ratio 16 \
  --skips 1 3 --save_pred --save_dir eval/bsergb_large
```

加 `--lpips` 计算 LPIPS；`--scenes basket_01 football_02` 可指定场景。
快速检查可以加 `--max_samples 2`，该限制对每种 skip 分别生效。

默认 `--spatial_mode official` 保留上游裁剪区域：BS-ERGB 中心裁到
`(H//32-1)*32 × (W//32-1)*32`；GOPRO 从左上裁到 32 的倍数。
`--spatial_mode pad` 则保留全图、右侧/下侧补齐后推理，再裁回原大小。
两种模式的评测区域不同，不能直接混报指标。

输出：

- 预测图：`eval/bsergb_large/<skip>/<scene>/<起始帧>_<目标帧>.png`
- 每帧指标：`frames.csv`
- 总体、按 skip 和按场景的均值：`metrics.json`

PSNR/SSIM 在 `[0,1]` 的浮点 RGB 上计算；LPIPS 使用上游 pyiqa 后端。
没有保留需要额外光流模型的 FloLPIPS。测试脚本与数据索引经过整理，未声称
已复现论文指标；对比实验应统一数据版本、裁剪、采样窗口和指标协议。

GOPRO 测试使用 `evaluate.py --dataset gopro`，其余参数相同；测试 x32 时使用
`--skips 31 --interp_ratio 32` 或 `--interp_ratio 0`。

## 7. 模型接口与单样本推理

```python
from models.model import MyModel

model = MyModel(voxel_bins=128, interp_ratio=16)
# im0/im1: [B,3,H,W]；events: [B,128,H,W]
# 输入 H/W 为 4 的倍数；times 为 [T] 或每行相同的 [B,T]
frames = model(im0, im1, events, [0.25, 0.5, 0.75])
# frames: [B,3,3,H,W]，依次为三个目标时间的 RGB 图像
```

不传时间时输出所有内部中间节点。传 `return_aux=True` 可返回最终两套
`flow_t0 / flow_t1`；它们是 MaskNet 修正后的目标时刻→端点光流。

```bash
python interpolate.py --checkpoint /path/to/Expv8_large.pt \
  --im0 /path/to/left.png --im1 /path/to/right.png \
  --voxel /path/to/voxel.npy --interp_ratio 16 \
  --times 0.25 0.5 0.75 --save_dir results
```

这里的体素应已经按对应数据预处理流程准备好，形状为 `[128,H,W]`；支持
NPY 或含 `data`/`voxel` 键的 NPZ。

## 8. 官方权重兼容与验证范围

严格支持以下权重容器：`model_state`、`model`、`state_dict` 或裸参数字典；
自动去除 `net.`、`module.` 前缀，保留网络内部参数名称。
使用 `strict=True`，不会通过忽略缺失参数来制造“加载成功”。

已做同一随机权重下与上游 Large 的 CPU 数值回归，覆盖 x2/x4/x8/x16/x32、
batch=2、内部 x16/实际输出 x4，输出最大绝对误差为 0。
这验证的是网络整理前后的计算一致性，并非真实官方权重的指标复现。
详细环境和检查结果见 `VALIDATION.md`。

运行必要检查：

```bash
python -m unittest discover -s tests -v
python tests/compare_upstream.py --upstream_root /path/to/original/TimeLens-XL
```

本次仅保留 Large；不支持将 Lights3、CBMNet、TimeLens 等其他模型的权重加载进来。
原始来源和使用条款说明见 `SOURCE_NOTICE.md`。

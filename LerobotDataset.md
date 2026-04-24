# LeRobotDataset 数据集结构详解

本文档详细介绍 LeRobotDataset 的目录结构、文件格式以及各 Parquet 文件中包含的字段含义。

---

## 1. 目录结构概览

```
LeRobotDataset/
├── meta/                          # 元数据目录
│   ├── info.json                  # 数据集基本信息（特征定义、fps等）
│   ├── stats.json                  # 归一化统计量（min/max/mean/std）
│   ├── tasks.parquet               # 任务描述表
│   └── episodes/                  # Episode边界信息
│       └── chunk-XXX/
│           └── file-XXX.parquet
├── videos/                        # 视频文件（可选，用于存储图像观测）
│   └── observation.images.{camera_name}/
│       └── chunk-XXX/
│           └── file-XXX.mp4
└── data/                         # 帧数据文件
    └── chunk-XXX/
        └── file-XXX.parquet
```

---

## 2. 核心概念

### Episode（回合）
- **定义**: 机器人执行一次完整任务的过程（一轮完整的动作序列）
- **示例**: "拿起粉色积木放进透明盒子" 就是一个 Episode
- **包含内容**: 多个连续的帧（Frame），从任务开始到任务结束

### Chunk（存储分片）
- **定义**: 为了优化存储和加载效率，将多个 Episode 的数据打包成的文件分片
- **与 Episode 的区别**: Chunk 是物理存储单元，Episode 是逻辑任务单元
- **数量关系**: 通常 1 个 Chunk 包含多个 Episode 的数据

### Frame（帧）
- **定义**: 数据采集的最小时间单元，对应机器人控制系统的一个时间步
- **采样率**: 由 `fps` 字段决定（如 30fps 表示每秒30帧）

---

## 3. info.json - 数据集元信息

**文件路径**: `meta/info.json`

**作用**: 定义数据集的基本信息、特征结构和存储路径模板。

```json
{
    "codebase_version": "v3.0",      // LeRobot代码库版本
    "robot_type": "so100_follower", // 机器人类型
    "total_episodes": 50,            // 总Episode数量
    "total_frames": 11939,           // 总帧数
    "total_tasks": 1,               // 任务类型数量
    "chunks_size": 1000,            // 每个Chunk包含的帧数（存储优化参数）
    "fps": 30,                      // 帧率（每秒帧数）
    "splits": {                     // 数据集划分
        "train": "0:50"             // 训练集：Episode 0 到 49
    },
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": { /* 详见3.1节 */ },
    "data_files_size_in_mb": 100,   // 数据文件大小（MB）
    "video_files_size_in_mb": 500  // 视频文件大小（MB）
}
```

### 3.1 features 特征定义

`features` 字段定义了数据集中所有特征的名称、类型和形状。

#### 动作特征 (action)

```json
"action": {
    "dtype": "float32",                              // 数据类型
    "shape": [6],                                     // 形状：6个关节
    "names": [                                        // 关节名称
        "shoulder_pan.pos",     // 肩部旋转
        "shoulder_lift.pos",    // 肩部升降
        "elbow_flex.pos",       // 肘部弯曲
        "wrist_flex.pos",       // 腕部弯曲
        "wrist_roll.pos",       // 腕部旋转
        "gripper.pos"           // 夹爪位置
    ],
    "fps": 30.0                                       // 采样率
}
```

#### 状态特征 (observation.state)

```json
"observation.state": {
    "dtype": "float32",
    "shape": [6],
    "names": [                                        // 与action相同
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos"
    ],
    "fps": 30.0
}
```

#### 图像特征 (observation.images.{camera_name})

```json
"observation.images.up": {
    "dtype": "video",                                // 类型为视频
    "shape": [480, 640, 3],                          // H x W x C
    "names": ["height", "width", "channels"],
    "info": {
        "video.height": 480,
        "video.width": 640,
        "video.codec": "av1",                        // 视频编码格式
        "video.pix_fmt": "yuv420p",                 // 像素格式
        "video.is_depth_map": false,                 // 是否为深度图
        "video.fps": 30,
        "video.channels": 3,
        "has_audio": false
    }
}
```

#### 索引特征

| 特征名 | dtype | shape | 说明 |
|--------|-------|-------|------|
| `timestamp` | float32 | [1] | 时间戳（秒） |
| `frame_index` | int64 | [1] | 帧索引（从0开始） |
| `episode_index` | int64 | [1] | Episode索引 |
| `index` | int64 | [1] | 全局索引 |
| `task_index` | int64 | [1] | 任务类型索引 |

---

## 4. stats.json - 归一化统计量

**文件路径**: `meta/stats.json`

**作用**: 存储每个特征的统计信息，用于数据归一化和反归一化。

### 4.1 数值特征的统计量

对于 `action` 和 `observation.state` 等数值特征：

```json
{
    "action": {
        "min": [-93.46, -100.0, 12.97, 33.53, -92.77, 0.0],
        "max": [88.01, 8.13, 100.0, 99.49, -20.0, 32.99],
        "mean": [8.02, -55.96, 65.26, 69.18, -53.42, 6.85],
        "std": [44.56, 36.49, 29.01, 13.24, 17.76, 9.0],
        "count": [11939]
    },
    "observation.state": {
        "min": [-92.65, -98.91, 16.81, 34.04, -92.28, 1.21],
        "max": [88.10, 8.79, 99.37, 98.64, -20.20, 32.68],
        "mean": [7.99, -55.18, 66.78, 69.25, -53.44, 8.11],
        "std": [44.49, 36.83, 27.71, 13.01, 17.70, 8.40],
        "count": [11939]
    }
}
```

### 4.2 图像特征的统计量

图像特征存储为 3D 张量（CHW 格式）的统计：

```json
{
    "observation.images.up": {
        "min": [[[0.0]], [[0.0]], [[0.0]]],        // 每个通道的最小值
        "max": [[[1.0]], [[1.0]], [[1.0]]],        // 每个通道的最大值
        "mean": [[[0.606]], [[0.614]], [[0.620]]], // 每个通道的均值
        "std": [[[0.146]], [[0.144]], [[0.140]]],  // 每个通道的标准差
        "count": [5000]                             // 采样帧数
    }
}
```

---

## 5. tasks.parquet - 任务描述表

**文件路径**: `meta/tasks.parquet`

**作用**: 存储每个任务类型的描述信息。

### 字段结构

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `task_index` | int64 | 任务索引 |
| `task` | string | 任务描述文本 |

### 示例数据

```
task_index  task
0           pink lego brick into the transparent box
```

---

## 6. episodes.parquet - Episode边界信息

**文件路径**: `meta/episodes/chunk-XXX/file-XXX.parquet`

**作用**: 存储每个 Episode 的起止帧索引、对应视频文件位置、以及统计信息。

### 核心字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `episode_index` | int64 | Episode编号 (0-49) |
| `data/chunk_index` | int64 | 数据文件所在的chunk索引 |
| `data/file_index` | int64 | 数据文件索引 |
| `dataset_from_index` | int64 | Episode起始帧在数据集中的索引 |
| `dataset_to_index` | int64 | Episode结束帧在数据集中的索引 |
| `videos/observation.images.{camera}/chunk_index` | int64 | 视频所在chunk索引 |
| `videos/observation.images.{camera}/file_index` | int64 | 视频文件索引 |
| `videos/observation.images.{camera}/from_timestamp` | float64 | 起始时间戳 |
| `videos/observation.images.{camera}/to_timestamp` | float64 | 结束时间戳 |
| `tasks` | string | 任务描述 |
| `length` | int64 | Episode包含的帧数 |

### Episode 与帧索引的对应关系

```
Episode 0: frame 0 到 frame 271  (272帧)
Episode 1: frame 303 到 frame 569 (267帧)
Episode 2: frame 569 到 frame 799 (231帧)
...
```

### 存储结构特点

episodes.parquet 文件采用**扁平化结构**，将每个 episode 的所有统计数据都存储在一行中，包括：
- 动作统计（min/max/mean/std）
- 状态统计
- 图像统计
- 时间戳统计
- 元信息

---

## 7. data.parquet - 帧数据内容

**文件路径**: `data/chunk-XXX/file-XXX.parquet`

**作用**: 存储每一帧的实际数据值。

### 字段结构

| 字段名 | 类型 | 形状 | 说明 |
|--------|------|------|------|
| `action` | object (ndarray) | [6] | 当前帧的关节动作值 |
| `observation.state` | object (ndarray) | [6] | 当前帧的关节状态值 |
| `timestamp` | float32 | [1] | 时间戳（秒） |
| `frame_index` | int64 | [1] | 帧索引 |
| `episode_index` | int64 | [1] | 所属Episode索引 |
| `index` | int64 | [1] | 全局索引 |
| `task_index` | int64 | [1] | 任务类型索引 |

### 数据示例

```python
# 第一帧数据 (index=0)
{
    'action': array([1.91, -99.41, 99.54, 74.84, -48.52, 1.62], dtype=float32),
    'observation.state': array([1.96, -98.74, 98.92, 74.82, -51.45, 1.41], dtype=float32),
    'timestamp': 0.0,
    'frame_index': 0,
    'episode_index': 0,
    'index': 0,
    'task_index': 0
}
```

### 存储格式说明

- `action` 和 `observation.state` 存储为 **object 类型**（实际是 numpy 数组的序列化形式）
- parquet 文件按 chunk 分组存储，每 chunk 约 1000 帧
- 视频帧数据不直接存储在 parquet 中，而是存储在 `videos/` 目录，通过 `episode_index` + `frame_index` 关联

---

## 8. videos/ - 视频文件

**目录路径**: `videos/observation.images.{camera_name}/`

**作用**: 存储摄像头采集的图像/视频数据。

### 目录结构

```
videos/
└── observation.images.up/           # 上方摄像头
    └── chunk-000/
        └── file-000.mp4            # 视频文件
└── observation.images.side/          # 侧方摄像头
    └── chunk-000/
        └── file-000.mp4
```

### 视频编码参数

- **codec**: av1 (AV1视频编码)
- **pix_fmt**: yuv420p
- **fps**: 30
- **channels**: 3 (RGB)

### 视频帧读取

视频帧通过 `frame_index` 计算时间戳来定位：
```python
timestamp = frame_index / fps  # 如 frame_index=0, fps=30 -> timestamp=0.0秒
```

---

## 9. 数据加载流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      LeRobotDataset 加载流程                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 读取 meta/info.json                                          │
│     └─ 获取 features、fps、chunks_size 等基础信息                │
│                                                                 │
│  2. 读取 meta/tasks.parquet                                      │
│     └─ 构建 task_index -> task 映射                              │
│                                                                 │
│  3. 读取 meta/episodes/*.parquet                                 │
│     └─ 获取每个 episode 的起止帧索引                             │
│     └─ 获取视频文件位置                                           │
│                                                                 │
│  4. 创建 EpisodeAwareSampler                                      │
│     └─ 按 episode 边界采样，确保 batch 不跨 episode              │
│                                                                 │
│  5. 读取 data/chunk-*/file-*.parquet                             │
│     └─ 按需加载指定帧的数据                                       │
│     └─ 通过 timestamp 关联视频帧                                   │
│                                                                 │
│  6. 应用 image_transforms (如 Resize, Normalize)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Delta Timestamps 机制

LeRobot 通过 `delta_timestamps` 支持**多帧历史/未来观察**：

```python
# SmolVLA 配置示例
observation_delta_indices = [0]           # 只用当前帧
action_delta_indices = list(range(50))    # 预测50个未来动作
```

```python
# 时间戳计算
delta_timestamps["observation.state"] = [0 / fps]        # [0.0] 秒
delta_timestamps["action"] = [i / fps for i in range(50)]  # [0.0, 0.033, ...] 秒
```

---

## 11. 快速参考

### 各文件用途总结

| 文件 | 用途 |
|------|------|
| `info.json` | 数据集元信息、特征定义 |
| `stats.json` | 归一化统计量 |
| `tasks.parquet` | 任务描述 |
| `episodes.parquet` | Episode边界和统计 |
| `data/*.parquet` | 帧数据（action、state等） |
| `videos/*.mp4` | 图像观测视频 |

### Episode 与 Chunk 的区别

| 属性 | Episode | Chunk |
|------|---------|-------|
| 含义 | 机器人执行一次任务 | 存储文件分片 |
| 数量 | 50个（示例数据集） | ~12个 |
| 用途 | 任务边界、episode采样 | 文件存储优化 |

### 数据类型速查

| 特征 | dtype | shape | 示例值 |
|------|-------|-------|--------|
| action | float32 | (6,) | [1.91, -99.41, 99.54, 74.84, -48.52, 1.62] |
| observation.state | float32 | (6,) | [1.96, -98.74, 98.92, 74.82, -51.45, 1.41] |
| observation.images.* | video | (H,W,3) | MP4视频文件 |
| timestamp | float32 | (1,) | 0.0 |
| frame_index | int64 | (1,) | 0 |
| episode_index | int64 | (1,) | 0 |
| task_index | int64 | (1,) | 0 |

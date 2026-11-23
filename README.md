# pairwise-doppler-3d-audio

## 领域

游戏、图形与多媒体

## 主题

3D 声源多普勒

## 说明

速度平滑、穿越奇点与原子读变换。

初始状态下 `media/doppler_3d_audio.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。

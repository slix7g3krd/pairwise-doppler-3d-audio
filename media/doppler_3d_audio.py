"""3D 声源多普勒：速度平滑、穿越奇点、原子变换与单链路衰减。

本模块对应六个缺陷修复：

* ``velocity_smooth``   —— 用时间窗内最小二乘拟合速度，替代逐帧差分。
* ``crossing_guard``    —— 声源接近/穿过 listener 时把多普勒因子拉回 1 并限幅。
* ``attenuation_dedupe``—— 距离衰减与空气吸收合并成一条链路，按标记去重。
* ``atomic_transform``  —— 读入完整不可变快照，绝不看到写了一半的世界变换。
* ``pause_history``     —— 暂停恢复时清空速度历史，避免恢复后持续偏移。
* ``doppler_cap``       —— 给多普勒因子一个显式上下限。

其余函数（``validation``、``audit``、``fallback``、``metrics``、``listener_move``、
``mix_thread``、``determinism``）是围绕上述修复的校验与混音辅助。
"""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

Vec3 = tuple[float, float, float]

SPEED_OF_SOUND = 343.0
DEFAULT_MIN_FACTOR = 0.5
DEFAULT_MAX_FACTOR = 2.0
DEFAULT_NEAR_EPSILON = 0.05
_EPS = 1e-9

DEFAULT_PARAMS: dict[str, float] = {
    "speed_of_sound": SPEED_OF_SOUND,
    "min_factor": DEFAULT_MIN_FACTOR,
    "max_factor": DEFAULT_MAX_FACTOR,
    "window": 0.1,
    "ref_distance": 1.0,
    "rolloff": 1.0,
    "min_distance": 0.1,
}


# ---------------------------------------------------------------------------
# 基础类型与数值工具
# ---------------------------------------------------------------------------

def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _vec(value: Any, default: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    """把数、序列、带 x/y/z 的对象或映射统一成 3 维向量。"""
    if value is None:
        return tuple(float(c) for c in default)  # type: ignore[return-value]
    if isinstance(value, bool):
        return tuple(float(c) for c in default)  # type: ignore[return-value]
    if isinstance(value, (int, float)):
        return (float(value), 0.0, 0.0)
    if isinstance(value, Mapping):
        for key in ("position", "pos", "xyz", "value"):
            if key in value:
                return _vec(value[key], default)
        return tuple(float(c) for c in default)  # type: ignore[return-value]
    xyz = getattr(value, "xyz", None)
    if xyz is not None:
        return _vec(xyz, default)
    if all(hasattr(value, axis) for axis in ("x", "y", "z")):
        return (
            _finite(getattr(value, "x"), 0.0),
            _finite(getattr(value, "y"), 0.0),
            _finite(getattr(value, "z"), 0.0),
        )
    try:
        sequence = list(value)
    except TypeError:
        return tuple(float(c) for c in default)  # type: ignore[return-value]
    coords = [_finite(item, 0.0) for item in sequence[:3]]
    while len(coords) < 3:
        coords.append(0.0)
    return (coords[0], coords[1], coords[2])


def _sub(left: Vec3, right: Vec3) -> Vec3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _dot(left: Vec3, right: Vec3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _length(value: Vec3) -> float:
    return math.sqrt(_dot(value, value))


def _ordered_bounds(min_factor: Any, max_factor: Any) -> tuple[float, float]:
    low = max(_finite(min_factor, DEFAULT_MIN_FACTOR), 0.0)
    high = max(_finite(max_factor, DEFAULT_MAX_FACTOR), 0.0)
    if high < low:
        low, high = high, low
    return low, high


def _clamp_factor(factor: Any, low: float, high: float) -> float:
    """限幅；NaN 回落到 1，±inf 落到对应的边界。"""
    try:
        value = float(factor)
    except (TypeError, ValueError):
        value = 1.0
    if math.isnan(value):
        value = 1.0
    elif value == math.inf:
        return high
    elif value == -math.inf:
        return low
    return min(max(value, low), high)


def _positive_int(value: Any, default: int = 1) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


@dataclass(frozen=True)
class Sample:
    """带时间戳的位置采样。"""

    time: float
    position: Vec3 = (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 1. 速度：时间窗平滑
# ---------------------------------------------------------------------------

def _coerce_samples(samples: Any) -> list[tuple[float, Vec3]]:
    """把各种样本写法统一成 ``(time, position)`` 列表。"""
    if samples is None or _is_scalar(samples):
        return []
    try:
        raw = list(samples)
    except TypeError:
        return []

    points: list[tuple[float, Vec3]] = []
    for index, item in enumerate(raw):
        if item is None:
            continue
        if _is_scalar(item):
            # 一串标量视为等间隔的 x 坐标。
            points.append((float(index), (float(item), 0.0, 0.0)))
            continue
        if isinstance(item, Mapping) and "position" in item:
            points.append((_finite(item.get("time", index), float(index)), _vec(item["position"])))
            continue
        position_attr = getattr(item, "position", None)
        if position_attr is not None and (hasattr(item, "time") or hasattr(item, "t")):
            time_attr = getattr(item, "time", getattr(item, "t", index))
            points.append((_finite(time_attr, float(index)), _vec(position_attr)))
            continue
        try:
            sequence = list(item)
        except TypeError:
            points.append((float(index), _vec(item)))
            continue
        if not sequence:
            continue
        # (time, vec3) —— 第二项不是标量。
        if len(sequence) == 2 and not _is_scalar(sequence[1]):
            points.append((_finite(sequence[0], float(index)), _vec(sequence[1])))
        # (time, x, y, z)
        elif len(sequence) == 4 and all(_is_scalar(value) for value in sequence):
            points.append((_finite(sequence[0], float(index)), _vec(sequence[1:])))
        # (x, y, z) 或 (x, y)
        else:
            points.append((float(index), _vec(sequence)))

    points.sort(key=lambda pair: pair[0])
    # 同一时刻只保留最后一次采样。
    deduped: list[tuple[float, Vec3]] = []
    for time_value, position in points:
        if deduped and abs(time_value - deduped[-1][0]) <= _EPS:
            deduped[-1] = (time_value, position)
        else:
            deduped.append((time_value, position))
    return deduped


def velocity_smooth(
    samples: Any,
    window: float = 0.1,
    *,
    default: Vec3 | None = None,
    max_speed: float | None = None,
) -> Vec3:
    """在最近的 ``window`` 秒内用最小二乘拟合速度。

    与逐帧差分不同，窗口内多个采样共同决定斜率，低帧率下的抖动会被平均掉。
    采样不足两个时返回 ``default``（默认零向量），而不是编造一个速度。
    """
    fallback = _vec(default) if default is not None else (0.0, 0.0, 0.0)
    points = _coerce_samples(samples)
    if len(points) < 2:
        return fallback

    span = max(_finite(window, 0.1), 0.0)
    if span > 0.0:
        cutoff = points[-1][0] - span
        selected = [pair for pair in points if pair[0] >= cutoff - _EPS]
        # 窗口内样本不够时，退回到最近的两个采样。
        if len(selected) < 2:
            selected = points[-2:]
    else:
        selected = points[-2:]

    count = len(selected)
    mean_t = sum(pair[0] for pair in selected) / count
    denominator = sum((pair[0] - mean_t) ** 2 for pair in selected)
    if denominator <= _EPS:
        return fallback

    mean_x = sum(pair[1][0] for pair in selected) / count
    mean_y = sum(pair[1][1] for pair in selected) / count
    mean_z = sum(pair[1][2] for pair in selected) / count
    vx = sum((pair[0] - mean_t) * (pair[1][0] - mean_x) for pair in selected) / denominator
    vy = sum((pair[0] - mean_t) * (pair[1][1] - mean_y) for pair in selected) / denominator
    vz = sum((pair[0] - mean_t) * (pair[1][2] - mean_z) for pair in selected) / denominator

    if max_speed is not None:
        limit = max(_finite(max_speed, 0.0), 0.0)
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        if speed > limit > 0.0:
            scale = limit / speed
            vx, vy, vz = vx * scale, vy * scale, vz * scale
        elif limit == 0.0:
            return (0.0, 0.0, 0.0)

    return (vx, vy, vz)


# ---------------------------------------------------------------------------
# 6. 多普勒因子上下限 + 2. 穿越 listener 的奇点保护
# ---------------------------------------------------------------------------

def doppler_cap(
    factor: Any,
    min_factor: float = DEFAULT_MIN_FACTOR,
    max_factor: float = DEFAULT_MAX_FACTOR,
) -> float:
    """把多普勒因子限制在 ``[min_factor, max_factor]`` 内。

    穿越 listener 时分母会过零，原始因子可能趋近 ±∞ 甚至变号；统一限幅后
    不会再出现正负之间瞬间跳变的爆音。
    """
    low, high = _ordered_bounds(min_factor, max_factor)
    return _clamp_factor(factor, low, high)


def crossing_guard(
    factor: Any,
    distance: Any = None,
    *,
    min_factor: float = DEFAULT_MIN_FACTOR,
    max_factor: float = DEFAULT_MAX_FACTOR,
    previous: Any = None,
    max_step: Any = None,
    near_epsilon: float = DEFAULT_NEAR_EPSILON,
) -> float:
    """处理声源接近/穿过 listener 的情况。

    * ``distance`` 小于 ``near_epsilon`` 时，把因子按距离权重拉回 1.0，
      因为此时视线方向本身不稳定，任何径向速度都不可信。
    * 最后始终经过 :func:`doppler_cap` 限幅。
    * 可选 ``previous``/``max_step``：限制单个混音块内的因子变化率。
    """
    low, high = _ordered_bounds(min_factor, max_factor)
    value = _clamp_factor(factor, low, high)

    epsilon = max(_finite(near_epsilon, DEFAULT_NEAR_EPSILON), 0.0)
    if distance is not None:
        try:
            separation = abs(float(distance))
        except (TypeError, ValueError):
            separation = math.inf
        if math.isnan(separation):
            separation = math.inf
        if epsilon > 0.0 and separation < epsilon:
            weight = separation / epsilon
            value = 1.0 + (value - 1.0) * weight
        elif epsilon == 0.0 and separation == 0.0:
            value = 1.0
        value = _clamp_factor(value, low, high)

    if previous is not None and max_step is not None:
        step = max(_finite(max_step, 0.0), 0.0)
        if step > 0.0:
            previous_value = _clamp_factor(previous, low, high)
            value = _clamp_factor(
                min(max(value, previous_value - step), previous_value + step),
                low,
                high,
            )
    return value


def _raw_doppler(
    snapshot: "TransformSnapshot",
    speed_of_sound: float,
    min_distance: float,
) -> tuple[float, float]:
    """返回 (未限幅的多普勒因子, 声源与 listener 的距离)。"""
    sound_speed = _finite(speed_of_sound, SPEED_OF_SOUND)
    if sound_speed <= 0.0:
        sound_speed = SPEED_OF_SOUND

    delta = _sub(snapshot.source_position, snapshot.listener_position)
    distance = _length(delta)
    floor = max(_finite(min_distance, 1e-4), 0.0)
    if distance <= floor or distance <= 0.0:
        return 1.0, distance

    direction = (delta[0] / distance, delta[1] / distance, delta[2] / distance)
    source_radial = _dot(snapshot.source_velocity, direction)
    listener_radial = _dot(snapshot.listener_velocity, direction)
    denominator = sound_speed + source_radial
    if abs(denominator) <= _EPS:
        return math.inf, distance
    return (sound_speed + listener_radial) / denominator, distance


def doppler_factor(
    source_position: Any,
    source_velocity: Any = (0.0, 0.0, 0.0),
    listener_position: Any = (0.0, 0.0, 0.0),
    listener_velocity: Any = (0.0, 0.0, 0.0),
    *,
    speed_of_sound: float = SPEED_OF_SOUND,
    min_distance: float = 1e-4,
    min_factor: float = DEFAULT_MIN_FACTOR,
    max_factor: float = DEFAULT_MAX_FACTOR,
    previous: Any = None,
    max_step: Any = None,
) -> float:
    """计算并保护单帧多普勒因子。"""
    snapshot = TransformSnapshot(
        sequence=0,
        source_position=_vec(source_position),
        listener_position=_vec(listener_position),
        source_velocity=_vec(source_velocity),
        listener_velocity=_vec(listener_velocity),
    )
    raw, distance = _raw_doppler(snapshot, speed_of_sound, min_distance)
    return crossing_guard(
        raw,
        distance,
        min_factor=min_factor,
        max_factor=max_factor,
        previous=previous,
        max_step=max_step,
        near_epsilon=max(_finite(min_distance, 1e-4), DEFAULT_NEAR_EPSILON),
    )


# ---------------------------------------------------------------------------
# 3. 衰减与空气吸收：单链路、去重
# ---------------------------------------------------------------------------

def distance_model(
    distance: Any,
    *,
    ref_distance: float = 1.0,
    rolloff: float = 1.0,
    min_distance: float = 0.1,
    max_gain: float = 1.0,
) -> float:
    """距离衰减增益，近场用 ``ref_distance`` 封顶避免距离趋零时发散。"""
    reference = max(_finite(ref_distance, 1.0), _finite(min_distance, 0.1), _EPS)
    floor = max(_finite(min_distance, 0.1), 0.0)
    separation = max(abs(_finite(distance, reference)), floor, _EPS)
    exponent = max(_finite(rolloff, 1.0), 0.0)
    gain = (reference / max(separation, reference)) ** exponent
    ceiling = max(_finite(max_gain, 1.0), 0.0)
    return float(min(gain, ceiling))


def _absorption_coefficient(
    frequency: float,
    temperature: float,
    humidity: float,
) -> float:
    """简化的空气吸收系数（Np/m），随频率与温湿度单调变化。

    这里不追求 ISO 9613-1 的精度，只保证量纲与趋势正确，并且始终非负。
    """
    hertz = max(_finite(frequency, 1000.0), 0.0)
    celsius = _finite(temperature, 20.0)
    relative_humidity = min(max(_finite(humidity, 50.0), 1.0), 100.0)
    temperature_factor = max(1.0 + 0.02 * (20.0 - celsius), 0.1)
    humidity_factor = 1.0 + abs(50.0 - relative_humidity) / 50.0
    return 5.8e-4 * (hertz / 1000.0) ** 1.6 * temperature_factor * humidity_factor


def air_absorb(
    distance: Any,
    *,
    frequency: float = 1000.0,
    temperature: float = 20.0,
    humidity: float = 50.0,
    coefficient: float | None = None,
    min_distance: float = 0.1,
) -> float:
    """空气吸收增益 ``exp(-alpha * path)``，``path`` 从 ``min_distance`` 起算。"""
    separation = abs(_finite(distance, 0.0))
    if coefficient is None:
        alpha = _absorption_coefficient(frequency, temperature, humidity)
    else:
        alpha = max(_finite(coefficient, 0.0), 0.0)
    path = max(separation - max(_finite(min_distance, 0.1), 0.0), 0.0)
    return float(math.exp(-alpha * path))


def attenuation_dedupe(
    distance: Any = 1.0,
    *,
    ref_distance: float = 1.0,
    rolloff: float = 1.0,
    min_distance: float = 0.1,
    max_gain: float = 1.0,
    frequency: float = 1000.0,
    temperature: float = 20.0,
    humidity: float = 50.0,
    coefficient: float | None = None,
    air_distance: Any = None,
    distance_applied: bool = False,
    absorption_applied: bool = False,
) -> dict[str, Any]:
    """把距离衰减与空气吸收合成一条链路，同一环节只应用一次。

    调用方若已在上游应用过某一环节（``distance_applied`` /
    ``absorption_applied``），该环节在这里被跳过而不是再乘一遍。
    返回字典中的 ``stages`` 记录实际执行过的环节，便于审计。
    """
    separation = abs(_finite(distance, _finite(ref_distance, 1.0)))
    stages: list[str] = []

    distance_gain = 1.0
    if not distance_applied:
        distance_gain = distance_model(
            separation,
            ref_distance=ref_distance,
            rolloff=rolloff,
            min_distance=min_distance,
            max_gain=max_gain,
        )
        stages.append("distance")

    absorption_gain = 1.0
    if not absorption_applied:
        path = separation if air_distance is None else abs(_finite(air_distance, separation))
        absorption_gain = air_absorb(
            path,
            frequency=frequency,
            temperature=temperature,
            humidity=humidity,
            coefficient=coefficient,
            min_distance=min_distance,
        )
        stages.append("air_absorption")

    return {
        "gain": float(distance_gain * absorption_gain),
        "distance_gain": float(distance_gain),
        "absorption_gain": float(absorption_gain),
        "stages": tuple(stages),
        "distance": float(separation),
        "distance_applied": bool(distance_applied),
        "absorption_applied": bool(absorption_applied),
    }


# ---------------------------------------------------------------------------
# 4. 世界变换：整体原子读
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformSnapshot:
    """一个版本内完整的世界变换；字段不可变，可安全跨线程共享。"""

    sequence: int = 0
    source_position: Vec3 = (0.0, 0.0, 0.0)
    listener_position: Vec3 = (0.0, 0.0, 0.0)
    source_velocity: Vec3 = (0.0, 0.0, 0.0)
    listener_velocity: Vec3 = (0.0, 0.0, 0.0)
    time: float = 0.0

    def __post_init__(self) -> None:
        # 允许调用方传入 list，但内部统一成 tuple 以保证真正不可变。
        for name in (
            "source_position",
            "listener_position",
            "source_velocity",
            "listener_velocity",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "source_position": self.source_position,
            "listener_position": self.listener_position,
            "source_velocity": self.source_velocity,
            "listener_velocity": self.listener_velocity,
            "time": self.time,
        }


class TransformBuffer:
    """写者整体替换快照，读者整体读取快照。

    异步混音线程拿到的 ``TransformSnapshot`` 一定来自同一个写入版本，
    不会出现位置已更新、速度仍是旧值这种“半更新”状态。
    """

    def __init__(self, initial: TransformSnapshot | None = None) -> None:
        self._snapshot = initial if isinstance(initial, TransformSnapshot) else TransformSnapshot()
        self._lock = threading.Lock()

    @property
    def sequence(self) -> int:
        return self._snapshot.sequence

    def read(self) -> TransformSnapshot:
        with self._lock:
            return self._snapshot

    def write(
        self,
        *,
        source_position: Any = None,
        listener_position: Any = None,
        source_velocity: Any = None,
        listener_velocity: Any = None,
        time: Any = None,
    ) -> TransformSnapshot:
        """在锁内基于旧快照构造新快照并一次性交换。"""
        with self._lock:
            old = self._snapshot
            new = TransformSnapshot(
                sequence=old.sequence + 1,
                source_position=old.source_position
                if source_position is None
                else _vec(source_position),
                listener_position=old.listener_position
                if listener_position is None
                else _vec(listener_position),
                source_velocity=old.source_velocity
                if source_velocity is None
                else _vec(source_velocity),
                listener_velocity=old.listener_velocity
                if listener_velocity is None
                else _vec(listener_velocity),
                time=old.time if time is None else _finite(time, old.time),
            )
            self._snapshot = new
            return new

    update = write


def _snapshot_from(
    source: Any = None,
    listener: Any = None,
    source_velocity: Any = None,
    listener_velocity: Any = None,
    time: Any = None,
) -> TransformSnapshot:
    source_position: Any = (0.0, 0.0, 0.0)
    listener_position: Any = (0.0, 0.0, 0.0)

    if isinstance(source, Mapping):
        source_position = source.get(
            "source_position", source.get("source", source.get("position"))
        )
        if source_velocity is None:
            source_velocity = source.get("source_velocity", source.get("velocity"))
        if listener is None:
            listener = source.get("listener_position", source.get("listener"))
    elif source is not None:
        if hasattr(source, "source_position") or hasattr(source, "position"):
            source_position = getattr(source, "source_position", None)
            if source_position is None:
                source_position = getattr(source, "position", None)
            if source_velocity is None:
                source_velocity = getattr(source, "source_velocity", None)
                if source_velocity is None:
                    source_velocity = getattr(source, "velocity", None)
        else:
            source_position = source

    if isinstance(listener, Mapping):
        listener_position = listener.get(
            "listener_position", listener.get("listener", listener.get("position"))
        )
        if listener_velocity is None:
            listener_velocity = listener.get(
                "listener_velocity", listener.get("velocity")
            )
    elif listener is not None:
        if hasattr(listener, "listener_position") or hasattr(listener, "position"):
            listener_position = getattr(listener, "listener_position", None)
            if listener_position is None:
                listener_position = getattr(listener, "position", None)
            if listener_velocity is None:
                listener_velocity = getattr(listener, "listener_velocity", None)
                if listener_velocity is None:
                    listener_velocity = getattr(listener, "velocity", None)
        else:
            listener_position = listener

    return TransformSnapshot(
        sequence=0,
        source_position=_vec(source_position),
        listener_position=_vec(listener_position),
        source_velocity=_vec(source_velocity),
        listener_velocity=_vec(listener_velocity),
        time=_finite(time, 0.0),
    )


def atomic_transform(
    source: Any = None,
    *,
    listener: Any = None,
    source_velocity: Any = None,
    listener_velocity: Any = None,
    time: Any = None,
    buffer: TransformBuffer | None = None,
) -> TransformSnapshot:
    """整体读取（或构造）一份世界变换快照。

    传入 :class:`TransformBuffer` 时只做一次 ``read()``，返回不可变快照；
    否则用本次参数构造快照。无论哪条路径，调用方拿到的都是完整版本。
    """
    if isinstance(source, TransformBuffer):
        return source.read()
    if isinstance(source, TransformSnapshot):
        return source
    if buffer is not None:
        return buffer.read()
    return _snapshot_from(
        source,
        listener,
        source_velocity,
        listener_velocity,
        time,
    )


# ---------------------------------------------------------------------------
# 5. 暂停/恢复：清空速度历史
# ---------------------------------------------------------------------------

class VelocityHistory:
    """可暂停的速度采样历史；恢复时丢弃暂停前的样本。"""

    def __init__(self, window: float = 0.1, maxlen: int = 512) -> None:
        self.window = max(_finite(window, 0.1), 0.0)
        self.samples: deque[tuple[float, Vec3]] = deque(maxlen=max(int(maxlen), 1))
        self.paused = False
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.samples)

    def clear(self) -> None:
        self.samples.clear()
        self.epoch += 1

    def add(self, time: Any, position: Any) -> bool:
        if self.paused:
            return False
        self.samples.append((_finite(time, 0.0), _vec(position)))
        return True

    def set_paused(self, paused: Any) -> bool:
        """切换暂停状态；返回是否因为恢复而清空了历史。"""
        paused = bool(paused)
        cleared = False
        if self.paused and not paused:
            # 暂停期间世界可能已变化，旧速度不能继续积分。
            self.clear()
            cleared = True
        self.paused = paused
        return cleared

    def velocity(self, *, max_speed: float | None = None) -> Vec3:
        return velocity_smooth(
            list(self.samples),
            self.window,
            max_speed=max_speed,
        )


def pause_history(
    history: Any = None,
    paused: bool = True,
    *,
    now: Any = None,
) -> dict[str, Any]:
    """暂停/恢复钩子；恢复（``paused=False``）时清空速度历史。

    支持 :class:`VelocityHistory`、普通 ``list``/``deque`` 或标量占位。
    """
    del now  # 兼容旧调用签名，目前不需要时间戳
    paused = bool(paused)

    if isinstance(history, VelocityHistory):
        cleared = history.set_paused(paused)
        return {
            "paused": history.paused,
            "history_size": len(history),
            "epoch": history.epoch,
            "cleared": cleared,
        }

    if isinstance(history, (list, deque)):
        cleared = not paused
        if cleared:
            history.clear()
        return {
            "paused": paused,
            "history_size": len(history),
            "epoch": 0,
            "cleared": cleared,
        }

    return {
        "paused": paused,
        "history_size": 0,
        "epoch": 0,
        "cleared": False,
    }


# ---------------------------------------------------------------------------
# 校验、审计、回退与指标
# ---------------------------------------------------------------------------

def validation(config: Any = None, **overrides: Any) -> dict[str, Any]:
    """校验参数；返回 ``{"ok", "errors", "config"}``。"""
    merged: dict[str, Any] = dict(DEFAULT_PARAMS)
    if isinstance(config, Mapping):
        merged.update(config)
    merged.update(overrides)

    errors: list[str] = []
    sound_speed = _finite(merged.get("speed_of_sound"), SPEED_OF_SOUND)
    if sound_speed <= 0.0:
        errors.append("speed_of_sound must be > 0")

    min_factor = _finite(merged.get("min_factor"), DEFAULT_MIN_FACTOR)
    max_factor = _finite(merged.get("max_factor"), DEFAULT_MAX_FACTOR)
    if min_factor <= 0.0:
        errors.append("min_factor must be > 0")
    if max_factor < min_factor:
        errors.append("max_factor must be >= min_factor")

    if _finite(merged.get("window"), 0.1) < 0.0:
        errors.append("window must be >= 0")
    if _finite(merged.get("ref_distance"), 1.0) <= 0.0:
        errors.append("ref_distance must be > 0")
    if _finite(merged.get("rolloff"), 1.0) < 0.0:
        errors.append("rolloff must be >= 0")
    if _finite(merged.get("min_distance"), 0.1) < 0.0:
        errors.append("min_distance must be >= 0")
    if not 0.0 <= _finite(merged.get("humidity", 50.0), 50.0) <= 100.0:
        errors.append("humidity must be within [0, 100]")

    return {"ok": not errors, "errors": errors, "config": merged}


def audit(
    events: Any = None,
    *,
    min_factor: float = DEFAULT_MIN_FACTOR,
    max_factor: float = DEFAULT_MAX_FACTOR,
) -> dict[str, Any]:
    """检查快照/事件流：非有限值、越界因子、重复吸收。"""
    low, high = _ordered_bounds(min_factor, max_factor)

    if events is None:
        items: list[Any] = []
    elif isinstance(events, (TransformSnapshot, Mapping, str, bytes)) or _is_scalar(events):
        items = [events]
    else:
        try:
            items = list(events)
        except TypeError:
            items = [events]

    issues: list[str] = []
    for index, event in enumerate(items):
        if isinstance(event, TransformSnapshot):
            for name in (
                "source_position",
                "listener_position",
                "source_velocity",
                "listener_velocity",
            ):
                values = getattr(event, name)
                if any(not math.isfinite(float(component)) for component in values):
                    issues.append(f"event {index}: {name} contains a non-finite value")
            if event.sequence < 0:
                issues.append(f"event {index}: negative sequence")
            continue

        if isinstance(event, Mapping):
            factor = event.get("factor", event.get("doppler_factor"))
            if factor is not None:
                value = _finite(factor, 1.0)
                if value < low or value > high:
                    issues.append(
                        f"event {index}: factor {value:.6g} outside [{low:g}, {high:g}]"
                    )
            stages = event.get("stages")
            if stages is not None:
                try:
                    stage_list = list(stages)
                except TypeError:
                    stage_list = []
                if stage_list.count("air_absorption") > 1:
                    issues.append(f"event {index}: air absorption applied more than once")
                if stage_list.count("distance") > 1:
                    issues.append(f"event {index}: distance attenuation applied more than once")
            if event.get("absorption_applied") and event.get("air_absorption_applied"):
                issues.append(f"event {index}: absorption applied twice")
            continue

        if _is_scalar(event):
            value = _finite(event, 1.0)
            if value < low or value > high:
                issues.append(
                    f"event {index}: factor {value:.6g} outside [{low:g}, {high:g}]"
                )
            continue

        # 其它可迭代的非法事件
        if event is not None:
            issues.append(f"event {index}: unrecognized event type {type(event).__name__}")

    return {"ok": not issues, "checked": len(items), "issues": issues}


def fallback(
    value: Any = None,
    *,
    default: Any = 0.0,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Any:
    """把无效输入回退成安全值；映射则补全并清洗默认参数。"""
    if isinstance(value, Mapping):
        merged: dict[str, Any] = dict(DEFAULT_PARAMS)
        merged.update(value)
        cleaned: dict[str, Any] = {}
        for key, item in merged.items():
            if _is_scalar(item):
                cleaned[key] = _finite(item, float(DEFAULT_PARAMS.get(key, 0.0)))
            else:
                cleaned[key] = item
        return cleaned

    if value is None:
        candidate = _finite(default, 0.0)
    else:
        candidate = _finite(value, _finite(default, 0.0))
    if minimum is not None:
        candidate = max(candidate, _finite(minimum, candidate))
    if maximum is not None:
        candidate = min(candidate, _finite(maximum, candidate))
    return candidate


def metrics(
    factors: Any = None,
    *,
    min_factor: float = DEFAULT_MIN_FACTOR,
    max_factor: float = DEFAULT_MAX_FACTOR,
) -> dict[str, Any]:
    """汇总多普勒因子序列的统计量。"""
    low, high = _ordered_bounds(min_factor, max_factor)

    if factors is None:
        raw: list[Any] = []
    elif _is_scalar(factors):
        raw = [factors]
    else:
        try:
            raw = list(factors)
        except TypeError:
            raw = [factors]

    values: list[float] = []
    for item in raw:
        if isinstance(item, Mapping):
            item = item.get("factor", item.get("doppler_factor", item.get("value")))
        if item is None or not _is_scalar(item):
            continue
        values.append(_finite(item, 1.0))

    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "clamped_low": 0,
            "clamped_high": 0,
            "min_factor": low,
            "max_factor": high,
        }

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "clamped_low": sum(1 for value in values if value <= low + _EPS),
        "clamped_high": sum(1 for value in values if value >= high - _EPS),
        "min_factor": low,
        "max_factor": high,
    }


# ---------------------------------------------------------------------------
# listener 更新与异步混音
# ---------------------------------------------------------------------------

def listener_move(
    target: Any = None,
    position: Any = None,
    velocity: Any = None,
    *,
    time: Any = None,
    buffer: TransformBuffer | None = None,
) -> Any:
    """原子更新 listener 的位置/速度。

    传入 :class:`TransformBuffer` 时只替换 listener 字段（位置与速度在同一个
    新快照里同时生效），返回更新后的完整快照；否则返回清洗后的 listener 状态。
    """
    active = buffer
    if active is None and isinstance(target, TransformBuffer):
        active = target

    if active is not None:
        current = active.read()
        new_position = current.listener_position if position is None else _vec(position)
        new_velocity = current.listener_velocity if velocity is None else _vec(velocity)
        return active.write(
            listener_position=new_position,
            listener_velocity=new_velocity,
            time=time,
        )

    if position is None and target is not None:
        position = target
    return {
        "listener_position": _vec(position),
        "listener_velocity": _vec(velocity),
        "time": _finite(time, 0.0),
    }


def mix_thread(
    buffer: Any = None,
    *,
    source_position: Any = None,
    source_velocity: Any = None,
    listener_position: Any = None,
    listener_velocity: Any = None,
    time: Any = None,
    blocks: int = 1,
    block_duration: float = 0.05,
    speed_of_sound: float = SPEED_OF_SOUND,
    min_factor: float = DEFAULT_MIN_FACTOR,
    max_factor: float = DEFAULT_MAX_FACTOR,
    max_step: Any = None,
    ref_distance: float = 1.0,
    rolloff: float = 1.0,
    min_distance: float = 0.1,
) -> dict[str, Any]:
    """模拟异步混音：每个块只读一次整体快照，再计算因子与单链路增益。"""
    del block_duration  # 快照已带时间；块时长仅作未来扩展

    if isinstance(buffer, TransformBuffer):
        transform_buffer = buffer
        count = _positive_int(blocks, 1)
        snapshots: list[int] = []
    else:
        count = _positive_int(buffer, 1) if _is_scalar(buffer) else _positive_int(blocks, 1)
        initial = _snapshot_from(
            None if _is_scalar(buffer) else buffer,
            listener_position,
            source_velocity,
            listener_velocity,
            time,
        )
        if source_position is not None:
            initial = TransformSnapshot(
                sequence=initial.sequence,
                source_position=_vec(source_position),
                listener_position=initial.listener_position,
                source_velocity=initial.source_velocity,
                listener_velocity=initial.listener_velocity,
                time=initial.time,
            )
        transform_buffer = TransformBuffer(initial)
        snapshots = []

    low, high = _ordered_bounds(min_factor, max_factor)
    factors: list[float] = []
    gains: list[float] = []
    distances: list[float] = []
    previous: float | None = None

    for _ in range(count):
        # 关键：每块只取一次快照，后续计算全部基于同一版本。
        snapshot = transform_buffer.read()
        raw, distance = _raw_doppler(snapshot, speed_of_sound, min_distance)
        factor = crossing_guard(
            raw,
            distance,
            min_factor=low,
            max_factor=high,
            previous=previous,
            max_step=max_step,
        )
        attenuation = attenuation_dedupe(
            distance,
            ref_distance=ref_distance,
            rolloff=rolloff,
            min_distance=min_distance,
        )
        factors.append(factor)
        gains.append(float(attenuation["gain"]))
        distances.append(distance)
        previous = factor
        snapshots.append(snapshot.sequence)

    return {
        "blocks": count,
        "factors": factors,
        "gains": gains,
        "distances": distances,
        "sequences": snapshots,
        "metrics": metrics(factors, min_factor=low, max_factor=high),
    }


# ---------------------------------------------------------------------------
# 确定性
# ---------------------------------------------------------------------------

def _canonical(value: Any) -> Any:
    if isinstance(value, TransformSnapshot):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    if _is_scalar(value):
        return _finite(value)
    return repr(value)


def determinism(
    target: Any = None,
    *,
    iterations: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """重复执行/读取并比较结果，用于发现非确定性的变换或混音。"""
    count = max(_positive_int(iterations, 3), 1)
    results: list[Any] = []
    for _ in range(count):
        if callable(target):
            results.append(_canonical(target(**kwargs)))
        elif isinstance(target, TransformBuffer):
            results.append(_canonical(target.read()))
        else:
            results.append(_canonical(target))

    first = results[0]
    return {
        "deterministic": all(result == first for result in results[1:]),
        "iterations": count,
        "signature": first,
    }


__all__ = [
    "SPEED_OF_SOUND",
    "DEFAULT_MIN_FACTOR",
    "DEFAULT_MAX_FACTOR",
    "DEFAULT_PARAMS",
    "Sample",
    "TransformSnapshot",
    "TransformBuffer",
    "VelocityHistory",
    "velocity_smooth",
    "crossing_guard",
    "doppler_cap",
    "doppler_factor",
    "attenuation_dedupe",
    "distance_model",
    "air_absorb",
    "atomic_transform",
    "pause_history",
    "validation",
    "audit",
    "fallback",
    "metrics",
    "listener_move",
    "mix_thread",
    "determinism",
]

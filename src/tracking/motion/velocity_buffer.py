"""Rolling position buffer cho tính velocity và acceleration (Plan.md §2.1)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np

from ..types import Point, Velocity

DEFAULT_MAX_SIZE: int = 10
DEFAULT_MIN_POINTS_VELOCITY: int = 2
DEFAULT_MIN_POINTS_ACCELERATION: int = 3


@dataclass
class VelocityBuffer:
    """Buffer rolling lưu lịch sử (vị trí, timestamp) của một track.

    Dùng cho ước lượng vận tốc và gia tốc bằng least-squares regression.

    Quy ước đơn vị:
        - ``position``     : pixels, dạng ``Point = Tuple[int, int]``
        - ``timestamp``    : seconds (epoch hoặc wall-clock), monotonic non-decreasing
        - ``velocity``     : pixels / second   (từ linear fit)
        - ``acceleration`` : pixels / second^2 (từ quadratic fit)

    Buffer chứa tối đa ``max_size`` mẫu gần nhất; mẫu cũ bị loại tự động
    thông qua ``deque(maxlen=...)``.
    """

    max_size: int = DEFAULT_MAX_SIZE
    min_points_for_velocity: int = DEFAULT_MIN_POINTS_VELOCITY
    min_points_for_acceleration: int = DEFAULT_MIN_POINTS_ACCELERATION

    timestamps: Deque[float] = field(init=False)
    positions: Deque[Point] = field(init=False)

    def __post_init__(self) -> None:
        if self.max_size < self.min_points_for_acceleration:
            raise ValueError(
                "max_size phải ≥ min_points_for_acceleration "
                f"(nhận được max_size={self.max_size}, "
                f"min_points_for_acceleration={self.min_points_for_acceleration})."
            )
        if self.min_points_for_velocity < 2:
            raise ValueError(
                "min_points_for_velocity phải ≥ 2 "
                f"(nhận được {self.min_points_for_velocity})."
            )
        self.timestamps = deque(maxlen=self.max_size)
        self.positions = deque(maxlen=self.max_size)

    def push(self, point: Point, timestamp: float) -> None:
        """Thêm một mẫu (vị trí, timestamp) vào buffer.

        Args:
            point:     Vị trí dạng ``(x, y)`` theo pixels.
            timestamp: Mốc thời gian theo giây; phải ≥ timestamp trước đó.

        Raises:
            ValueError: Nếu ``timestamp`` nhỏ hơn timestamp gần nhất (non-monotonic).
        """
        if self.timestamps and timestamp < self.timestamps[-1]:
            raise ValueError(
                "timestamp phải monotonic non-decreasing "
                f"(nhận được {timestamp}, trước đó {self.timestamps[-1]})."
            )
        self.timestamps.append(float(timestamp))
        self.positions.append((int(point[0]), int(point[1])))

    def clear(self) -> None:
        """Xóa toàn bộ mẫu đang lưu."""
        self.timestamps.clear()
        self.positions.clear()

    def __len__(self) -> int:
        return len(self.timestamps)

    def get_velocity(self) -> Optional[Velocity]:
        """Ước lượng vận tốc (vx, vy) theo pixels/second trên toàn buffer.

        Returns:
            ``(vx, vy)`` nếu có ≥ ``min_points_for_velocity`` mẫu và khoảng
            thời gian khác 0; ngược lại ``None``.
        """
        return self._fit_velocity(self._array_slice(len(self.timestamps)))

    def get_smoothed_velocity(self, window: int = 3) -> Optional[Velocity]:
        """Vận tốc regression trên ``window`` mẫu gần nhất (smoothing).

        Cửa sổ hẹp phản ứng nhanh hơn với thay đổi gần đây so với
        ``get_velocity()`` chạy trên toàn buffer.

        Args:
            window: Số mẫu cuối dùng cho hồi quy (≥ ``min_points_for_velocity``).

        Returns:
            ``(vx, vy)`` hoặc ``None`` nếu buffer không đủ mẫu.

        Raises:
            ValueError: Nếu ``window`` nhỏ hơn ``min_points_for_velocity``.
        """
        if window < self.min_points_for_velocity:
            raise ValueError(
                "window phải ≥ min_points_for_velocity "
                f"(nhận được window={window}, "
                f"min_points_for_velocity={self.min_points_for_velocity})."
            )
        length = min(window, len(self.timestamps))
        return self._fit_velocity(self._array_slice(length))

    def get_acceleration(self) -> Optional[Velocity]:
        """Ước lượng gia tốc (ax, ay) theo pixels/second^2 bằng quadratic fit.

        Returns:
            ``(ax, ay)`` nếu có ≥ ``min_points_for_acceleration`` mẫu và
            khoảng thời gian khác 0; ngược lại ``None``.
        """
        length = len(self.timestamps)
        if length < self.min_points_for_acceleration:
            return None

        ts, xs, ys = self._array_slice(length)
        if ts[-1] == ts[0]:
            return None
        if np.unique(ts).size < 3:
            return None

        # Dịch mốc thời gian về 0 để tránh RankWarning với timestamp epoch lớn.
        fit_ts = ts - ts[0]

        # polyfit bậc 2: y = a*t^2 + b*t + c  →  acc = 2a
        ax = 2.0 * float(np.polyfit(fit_ts, xs, 2)[0])
        ay = 2.0 * float(np.polyfit(fit_ts, ys, 2)[0])
        return (ax, ay)

    # ─── Internal helpers ────────────────────────────────────────────────

    def _array_slice(
        self, length: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Trả về (ts, xs, ys) là numpy array của ``length`` mẫu cuối."""
        if length <= 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty, empty

        # Lấy ``length`` mẫu cuối, tách thành mảng timestamp, tọa độ x và y.
        ts_list = list(self.timestamps)[-length:]
        pos_list = list(self.positions)[-length:]
        ts = np.asarray(ts_list, dtype=np.float64)
        pos = np.asarray(pos_list, dtype=np.float64)
        xs = pos[:, 0]
        ys = pos[:, 1]
        return ts, xs, ys

    def _fit_velocity(
        self, arrays: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Optional[Velocity]:
        """Ước lượng (vx, vy) bằng hồi quy bậc 1 (least-squares) theo thời gian."""
        ts, xs, ys = arrays
        # Cần tối thiểu số mẫu; nếu mọi timestamp trùng nhau thì chia cho dt=0 → bỏ.
        if ts.size < self.min_points_for_velocity:
            return None
        if ts[-1] == ts[0]:
            return None
        if np.unique(ts).size < 2:
            return None

        # Dịch mốc thời gian về 0 để tránh RankWarning với timestamp epoch lớn.
        fit_ts = ts - ts[0]

        # polyfit bậc 1: x = vx*t + b  →  hệ số bậc nhất (slope) chính là vận tốc.
        vx = float(np.polyfit(fit_ts, xs, 1)[0])
        vy = float(np.polyfit(fit_ts, ys, 1)[0])
        return (vx, vy)


__all__ = ["VelocityBuffer"]

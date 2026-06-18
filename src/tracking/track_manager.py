from __future__ import annotations

import time
from typing import Callable, List, Protocol, Sequence

from .kalman_filter import BoundingBoxKalmanFilter
from .motion.velocity_buffer import VelocityBuffer
from .matching import match_tracks_detections
from .types import Point, Track, TrackStatus


class TrackDetection(Protocol):
    """Giao diện tối thiểu mà một detection cần có để TrackManager xử lý."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    anchor_point: Point | None
    in_roi: bool
    zone_name: str | None
    risk_level: str | None
    distance_m: float | None


class TrackManager:
    """Quản lý vòng đời track bằng phép gán dựa trên IoU."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_misses: int = 5,
        min_hits: int = 2,
        max_trace_length: int = 30,
        time_provider: Callable[[], float] | None = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.max_trace_length = max_trace_length
        self._time_provider = time_provider or time.time

        self.tracks: List[Track] = []
        self._next_track_id = 1

    def update(self, detections: Sequence[TrackDetection]) -> List[Track]:
        """Cập nhật các track đang hoạt động từ detection của frame hiện tại."""

        # Bước dự đoán (prediction): mỗi track tự đẩy bbox về vị trí dự kiến
        # ở frame này dựa trên Kalman, TRƯỚC khi so khớp với detection mới.
        self._increment_track_ages()

        # So khớp track ↔ detection theo IoU (Hungarian matching).
        matches, unmatched_tracks, unmatched_detections = match_tracks_detections(
            self.tracks,
            detections,
            iou_threshold=self.iou_threshold,
        )

        # Track khớp được detection → cập nhật (bước hiệu chỉnh/correction của Kalman).
        for track_idx, detection_idx in matches:
            self._update_track(self.tracks[track_idx], detections[detection_idx])

        # Track không khớp detection nào → đánh dấu là bị mất (missed) frame này.
        for track_idx in unmatched_tracks:
            self._mark_missed(self.tracks[track_idx])

        # Detection không khớp track nào → khởi tạo track mới.
        for detection_idx in unmatched_detections:
            self.tracks.append(self._create_track(detections[detection_idx]))

        # Loại bỏ những track mất quá lâu (vượt max_misses).
        self._prune_dead_tracks()
        return list(self.tracks)

    def reset(self) -> None:
        """Xóa toàn bộ track đang hoạt động và đặt lại bộ cấp ID."""
        self.tracks.clear()
        self._next_track_id = 1

    def _increment_track_ages(self) -> None:
        """Tăng tuổi track và chạy bước dự đoán Kalman cho mỗi track.

        Đây là bước "prediction" của bộ lọc Kalman: dùng mô hình chuyển động
        (vận tốc không đổi) để ngoại suy vị trí bbox sang frame hiện tại khi
        chưa có quan sát mới. Nhờ đó việc so khớp IoU ngay sau đó chính xác hơn
        và track không bị "đứng yên" khi vật thể đang di chuyển.
        """
        for track in self.tracks:
            track.age += 1
            if track.kalman is not None:
                # predict() trả về bbox đã ngoại suy; get_velocity() lấy (vx, vy)
                # hiện tại từ vector trạng thái Kalman.
                track.bbox = track.kalman.predict()
                track.velocity = track.kalman.get_velocity()

    def _create_track(self, detection: TrackDetection) -> Track:
        """Khởi tạo một track mới từ detection chưa được gán."""
        # Nếu min_hits <= 1 thì track được xác nhận ngay; ngược lại còn TENTATIVE
        # (chờ đủ số lần khớp liên tiếp mới chuyển sang CONFIRMED).
        status = (
            TrackStatus.CONFIRMED
            if self.min_hits <= 1
            else TrackStatus.TENTATIVE
        )
        # Mỗi track sở hữu một bộ lọc Kalman riêng; initiate() nạp bbox quan sát
        # đầu tiên làm trạng thái khởi đầu (vận tốc khởi tạo bằng 0).
        kalman = BoundingBoxKalmanFilter()
        initial_bbox = kalman.initiate(detection.bbox)
        track = Track(
            track_id=self._next_track_id,
            bbox=initial_bbox,
            confidence=detection.confidence,
            class_id=detection.class_id,
            class_name=detection.class_name,
            age=1,
            hits=1,
            misses=0,
            is_confirmed=self.min_hits <= 1,
            status=status,
            velocity=kalman.get_velocity(),
            kalman=kalman,
            anchor_point=detection.anchor_point,
            in_roi=detection.in_roi,
            zone_name=detection.zone_name,
            risk_level=detection.risk_level,
            distance_m=detection.distance_m,
        )

        # Khởi tạo velocity_buffer để phục vụ ước lượng vận tốc/gia tốc và
        # dự đoán quỹ đạo (motion prediction) ở các frame sau.
        timestamp = float(self._time_provider())
        if detection.anchor_point is not None:
            track.velocity_buffer = VelocityBuffer()
            track.velocity_buffer.push(detection.anchor_point, timestamp)

        self._append_trace(track, detection.anchor_point)

        self._next_track_id += 1
        return track

    def _update_track(self, track: Track, detection: TrackDetection) -> None:
        """Hiệu chỉnh track bằng detection vừa khớp (bước correction của Kalman)."""
        # Phòng thủ: nếu vì lý do nào đó track chưa có Kalman thì tạo mới và nạp bbox.
        if track.kalman is None:
            track.kalman = BoundingBoxKalmanFilter()
            track.kalman.initiate(track.bbox)

        # update() kết hợp dự đoán trước đó với quan sát mới → bbox đã làm mượt.
        track.bbox = track.kalman.update(detection.bbox)
        track.confidence = detection.confidence
        track.class_id = detection.class_id
        track.class_name = detection.class_name

        track.anchor_point = detection.anchor_point
        track.in_roi = detection.in_roi
        track.zone_name = detection.zone_name
        track.risk_level = detection.risk_level
        track.distance_m = detection.distance_m

        # Khớp thành công frame này → tăng hits, reset misses về 0.
        track.hits += 1
        track.misses = 0
        track.is_confirmed = track.hits >= self.min_hits
        track.status = (
            TrackStatus.CONFIRMED
            if track.is_confirmed
            else TrackStatus.TENTATIVE
        )
        # Vận tốc tạm thời lấy từ Kalman (vx, vy).
        track.velocity = track.kalman.get_velocity()

        timestamp = float(self._time_provider())
        self._update_velocity_buffer(track, detection.anchor_point, timestamp)

        # Khi đã đủ lịch sử (≥5 mẫu), ưu tiên vận tốc làm mượt từ velocity_buffer
        # (hồi quy least-squares) vì ổn định hơn vận tốc tức thời của Kalman.
        if track.velocity_buffer is not None and len(track.velocity_buffer) >= 5:
            buf_vel = track.velocity_buffer.get_smoothed_velocity()
            if buf_vel is not None:
                track.velocity = buf_vel

        self._append_trace(track, detection.anchor_point)

    def _update_velocity_buffer(
        self,
        track: Track,
        point: Point | None,
        timestamp: float,
    ) -> None:
        """Đẩy mẫu (vị trí, timestamp) mới vào buffer phục vụ ước lượng vận tốc."""
        # Không có anchor_point thì bỏ qua, không làm bẩn buffer.
        if point is None:
            return

        if track.velocity_buffer is None:
            track.velocity_buffer = VelocityBuffer()

        track.velocity_buffer.push(point, timestamp)

    def _mark_missed(self, track: Track) -> None:
        """Đánh dấu track không khớp detection nào ở frame này (tăng misses)."""
        track.misses += 1
        track.status = TrackStatus.LOST

    def _prune_dead_tracks(self) -> None:
        """Loại bỏ các track đã mất quá lâu (misses vượt ngưỡng max_misses)."""
        self.tracks = [
            track for track in self.tracks if track.misses <= self.max_misses
        ]

    def _append_trace(self, track: Track, point: Point | None) -> None:
        """Ghi điểm anchor vào vệt di chuyển (trace), giới hạn độ dài tối đa."""
        if point is None:
            return

        track.trace.append(point)
        # Giữ trace không dài quá max_trace_length: chỉ lưu các điểm gần nhất.
        if len(track.trace) > self.max_trace_length:
            track.trace = track.trace[-self.max_trace_length :]


__all__ = ["TrackDetection", "TrackManager"]

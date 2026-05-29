from dataclasses import dataclass
from collections import deque
import time
import numpy as np


@dataclass
class SystemState:
    # 畫面變化程度，來自 motion estimator
    motion_score: float

    # 分類器不確定性
    # 0.0 = 很確定
    # 1.0 = 很不確定
    classifier_entropy: float = 0.0

    # tracker 信心
    # 1.0 = 很穩
    # 0.0 = 很不穩
    tracker_confidence: float = 1.0

    # detector 預算壓力
    # 0.0 = 很空
    # 1.0 = 快爆了
    budget_pressure: float = 0.0

    # 目前人車配對數，沒有就維持 0
    pair_count: int = 0

    # 與上次狀態相比的數量變化，沒有就維持 0
    count_change: float = 0.0


class BOSScheduler:

    def __init__(
        self,
        base_motion_threshold=0.03,
        history_size=20,
        urgency_threshold=0.40,
        force_refresh=5,
        burst_weight=1.5,

        # 每個時間窗最多呼叫幾次 detector
        max_calls_per_window=5,

        # 時間窗長度，單位秒
        window_seconds=1.0,

        # 防止剛 detect 完又連續 detect
        min_detect_interval=1,

        # 絕對最大 cache 年齡，避免 cache 變殭屍
        max_cache_age=None,
    ):

        self.base_motion_threshold = base_motion_threshold

        self.motion_history = deque(
            maxlen=history_size
        )

        self.urgency_threshold = urgency_threshold

        self.force_refresh = force_refresh

        self.burst_weight = burst_weight

        self.max_calls_per_window = max_calls_per_window

        self.window_seconds = window_seconds

        self.min_detect_interval = min_detect_interval

        if max_cache_age is None:
            self.max_cache_age = force_refresh * 2
        else:
            self.max_cache_age = max_cache_age

        # ==========================================
        # scheduler 自己管理 cache age
        # ==========================================
        self.cache_age = 0

        # ==========================================
        # detector budget window
        # ==========================================
        self.detect_timestamps = deque()

        # ==========================================
        # detect cooldown
        # ==========================================
        self.frames_since_detect = 999999

        # ==========================================
        # pair count state
        # ==========================================
        self.last_pair_count = 0

        # ==========================================
        # metrics
        # ==========================================
        self.total_frames = 0
        self.detect_calls = 0
        self.reuse_calls = 0
        self.budget_blocked_calls = 0

        # ==========================================
        # debug info
        # ==========================================
        self.last_urgency = 0.0

        self.last_dynamic_threshold = base_motion_threshold

        self.last_budget_pressure = 0.0

        self.last_reason = "init"
        self.start_time = time.time()

    def reset(self):

        self.motion_history.clear()

        self.cache_age = 0

        self.detect_timestamps.clear()

        self.frames_since_detect = 999999

        self.last_pair_count = 0

        self.total_frames = 0
        self.detect_calls = 0
        self.reuse_calls = 0
        self.budget_blocked_calls = 0

        self.last_urgency = 0.0

        self.last_dynamic_threshold = (
            self.base_motion_threshold
        )

        self.last_budget_pressure = 0.0

        self.last_reason = "reset"
        self.start_time = time.time()

    def update_motion_history(
        self,
        motion_score
    ):

        self.motion_history.append(
            float(motion_score)
        )

    def cleanup_budget_window(self):

        now = time.time()

        while self.detect_timestamps:

            if now - self.detect_timestamps[0] > self.window_seconds:

                self.detect_timestamps.popleft()

            else:

                break

    def count_calls_in_last(self, seconds: float):

        """Count detector calls in the last `seconds` seconds without mutating deque."""

        if not self.detect_timestamps:
            return 0

        now = time.time()

        cutoff = now - float(seconds)

        # deque is ordered oldest->newest; find first index >= cutoff
        # simple linear scan is fine for small windows
        cnt = 0
        for ts in reversed(self.detect_timestamps):
            if ts >= cutoff:
                cnt += 1
            else:
                break

        return cnt

    def compute_budget_pressure(self):

        self.cleanup_budget_window()

        if self.max_calls_per_window <= 0:

            self.last_budget_pressure = 1.0

            return 1.0

        pressure = len(self.detect_timestamps) / self.max_calls_per_window

        pressure = float(
            np.clip(
                pressure,
                0.0,
                1.0
            )
        )

        self.last_budget_pressure = pressure

        return pressure

    def can_call_detector(self):

        self.cleanup_budget_window()

        return len(self.detect_timestamps) < self.max_calls_per_window

    def record_detector_call(self):

        self.cleanup_budget_window()

        self.detect_timestamps.append(
            time.time()
        )

    def compute_dynamic_motion_threshold(self):

        if len(self.motion_history) < 3:

            self.last_dynamic_threshold = (
                self.base_motion_threshold
            )

            return self.base_motion_threshold

        avg_motion = float(
            np.mean(self.motion_history)
        )

        std_motion = float(
            np.std(self.motion_history)
        )

        dynamic_threshold = (
            avg_motion +
            self.burst_weight * std_motion
        )

        dynamic_threshold = max(
            dynamic_threshold,
            self.base_motion_threshold
        )

        self.last_dynamic_threshold = float(
            dynamic_threshold
        )

        return float(dynamic_threshold)

    def normalize_motion(
        self,
        motion_score
    ):

        dynamic_threshold = (
            self.compute_dynamic_motion_threshold()
        )

        normalized_motion = (
            float(motion_score) /
            max(dynamic_threshold, 1e-6)
        )

        normalized_motion = float(
            np.clip(
                normalized_motion,
                0.0,
                2.0
            )
        )

        return normalized_motion

    def compute_count_change(
        self,
        pair_count
    ):

        diff = abs(
            int(pair_count) -
            int(self.last_pair_count)
        )

        # 避免一台車進出就把 urgency 炸太高
        return float(
            np.clip(
                diff / 5.0,
                0.0,
                1.0
            )
        )

    def compute_staleness(
        self,
        state: SystemState
    ):

        motion_term = self.normalize_motion(
            state.motion_score
        )

        cache_term = min(
            self.cache_age / max(self.force_refresh, 1),
            1.0
        )

        tracker_risk = 1.0 - float(
            np.clip(
                state.tracker_confidence,
                0.0,
                1.0
            )
        )

        if state.count_change > 0:
            count_term = float(
                np.clip(
                    state.count_change,
                    0.0,
                    1.0
                )
            )
        else:
            count_term = self.compute_count_change(
                state.pair_count
            )

        staleness = (
            0.40 * cache_term +
            0.30 * motion_term +
            0.20 * count_term +
            0.10 * tracker_risk
        )

        return float(
            np.clip(
                staleness,
                0.0,
                1.5
            )
        )

    def compute_urgency(
        self,
        state: SystemState
    ):

        # ==========================================
        # 1. motion urgency
        # ==========================================
        motion_term = self.normalize_motion(
            state.motion_score
        )

        # ==========================================
        # 2. cache age urgency
        # ==========================================
        cache_term = min(
            self.cache_age / max(self.force_refresh, 1),
            1.0
        )

        # ==========================================
        # 3. tracker risk
        # ==========================================
        tracker_risk = 1.0 - float(
            np.clip(
                state.tracker_confidence,
                0.0,
                1.0
            )
        )

        # ==========================================
        # 4. classifier uncertainty
        # ==========================================
        entropy_term = float(
            np.clip(
                state.classifier_entropy,
                0.0,
                1.0
            )
        )

        # ==========================================
        # 5. count change
        # ==========================================
        if state.count_change > 0:

            count_term = float(
                np.clip(
                    state.count_change,
                    0.0,
                    1.0
                )
            )

        else:

            count_term = self.compute_count_change(
                state.pair_count
            )

        # ==========================================
        # 6. staleness
        # cache 不只看時間，也看 motion / count / tracker
        # ==========================================
        staleness_term = self.compute_staleness(
            state
        )

        # ==========================================
        # 7. budget pressure
        # 預算壓力越高，越不應該亂跑 YOLO
        # ==========================================
        internal_budget_pressure = self.compute_budget_pressure()

        external_budget_pressure = float(
            np.clip(
                state.budget_pressure,
                0.0,
                1.0
            )
        )

        budget_term = max(
            internal_budget_pressure,
            external_budget_pressure
        )

        # ==========================================
        # BOS-inspired urgency score
        # ==========================================
        urgency = (
            0.30 * motion_term +
            0.25 * cache_term +
            0.15 * staleness_term +
            0.10 * count_term +
            0.10 * entropy_term +
            0.05 * tracker_risk -
            0.15 * budget_term
        )

        urgency = float(
            np.clip(
                urgency,
                0.0,
                1.5
            )
        )

        self.last_urgency = urgency

        return urgency

    def should_detect(
        self,
        state: SystemState
    ):

        self.total_frames += 1

        self.update_motion_history(
            state.motion_score
        )

        urgency = self.compute_urgency(
            state
        )

        can_detect = self.can_call_detector()

        cooldown_active = (
            self.frames_since_detect <
            self.min_detect_interval
        )

        # ==========================================
        # cache 過舊，最高優先級
        # ==========================================
        if self.cache_age >= self.max_cache_age:

            if can_detect:

                self.cache_age = 0

                self.frames_since_detect = 0

                self.detect_calls += 1

                self.record_detector_call()

                self.last_pair_count = int(
                    state.pair_count
                )

                self.last_reason = "max_cache_age"

                return True, urgency, "max_cache_age"

            self.cache_age += 1

            self.frames_since_detect += 1

            self.reuse_calls += 1

            self.budget_blocked_calls += 1

            self.last_reason = "budget_blocked_max_cache"

            return False, urgency, "budget_blocked_max_cache"

        # ==========================================
        # 一般 force refresh
        # ==========================================
        if (
            self.cache_age >= self.force_refresh and
            not cooldown_active
        ):

            if can_detect:

                self.cache_age = 0

                self.frames_since_detect = 0

                self.detect_calls += 1

                self.record_detector_call()

                self.last_pair_count = int(
                    state.pair_count
                )

                self.last_reason = "force_refresh"

                return True, urgency, "force_refresh"

            self.cache_age += 1

            self.frames_since_detect += 1

            self.reuse_calls += 1

            self.budget_blocked_calls += 1

            self.last_reason = "budget_blocked_force_refresh"

            return False, urgency, "budget_blocked_force_refresh"

        # ==========================================
        # urgency 達標，跑 YOLO
        # ==========================================
        if (
            urgency >= self.urgency_threshold and
            not cooldown_active
        ):

            if can_detect:

                self.cache_age = 0

                self.frames_since_detect = 0

                self.detect_calls += 1

                self.record_detector_call()

                self.last_pair_count = int(
                    state.pair_count
                )

                self.last_reason = "high_urgency"

                return True, urgency, "high_urgency"

            self.cache_age += 1

            self.frames_since_detect += 1

            self.reuse_calls += 1

            self.budget_blocked_calls += 1

            self.last_reason = "budget_blocked_high_urgency"

            return False, urgency, "budget_blocked_high_urgency"

        # ==========================================
        # 否則沿用 cache
        # ==========================================
        self.cache_age += 1

        self.frames_since_detect += 1

        self.reuse_calls += 1

        self.last_reason = "reuse_cache"

        return False, urgency, "reuse_cache"

    def get_metrics(self):

        if self.total_frames <= 0:

            detect_ratio = 0.0
            reuse_ratio = 0.0

        else:

            detect_ratio = self.detect_calls / self.total_frames
            reuse_ratio = self.reuse_calls / self.total_frames
        # Use sliding-window counts for more responsive rates
        # per-second: count in last `window_seconds` seconds
        window = max(float(self.window_seconds), 0.001)
        calls_in_window = self.count_calls_in_last(window)
        detector_calls_per_second = float(calls_in_window) / window

        # per-minute: count in last 60s (without mutating deque)
        calls_last_min = self.count_calls_in_last(60.0)
        detector_calls_per_minute = float(calls_last_min) / 60.0

        return {
            "total_frames": self.total_frames,
            "detect_calls": self.detect_calls,
            "reuse_calls": self.reuse_calls,
            "budget_blocked_calls": self.budget_blocked_calls,
            "detect_ratio": float(detect_ratio),
            "reuse_ratio": float(reuse_ratio),
            "cache_age": self.cache_age,
            "frames_since_detect": self.frames_since_detect,
            "last_urgency": self.last_urgency,
            "last_dynamic_threshold": self.last_dynamic_threshold,
            "last_budget_pressure": self.last_budget_pressure,
            "last_reason": self.last_reason,
            "detector_calls_per_second": detector_calls_per_second,
            "detector_calls_per_minute": detector_calls_per_minute,
        }


# ==========================================
# 舊版相容包裝
# 若某些舊程式還在呼叫：
# scheduler.should_detect(motion_score)
# 可以繼續跑
# ==========================================
class MotionScheduler(BOSScheduler):

    def __init__(
        self,
        motion_threshold=0.03,
        max_idle=10,
        history_size=10,
        burst_weight=1.5
    ):

        super().__init__(
            base_motion_threshold=motion_threshold,
            history_size=history_size,
            urgency_threshold=0.55,
            force_refresh=max_idle,
            burst_weight=burst_weight,
            max_cache_age=max_idle * 2,
        )

    def should_detect(
        self,
        motion_score
    ):

        state = SystemState(
            motion_score=motion_score,
            classifier_entropy=0.0,
            tracker_confidence=1.0,
            budget_pressure=0.0,
            pair_count=0,
            count_change=0.0,
        )

        run_detect, urgency, reason = super().should_detect(
            state
        )

        return run_detect
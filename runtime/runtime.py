import time
from dataclasses import dataclass, field


@dataclass
class RuntimeState:

    cached_detections: list = field(default_factory=list)

    violation_count: int = 0

    violation_frame_count: int = 0

    violation_active: bool = False

    yolo_history: list = field(default_factory=list)

    yolo_calls: int = 0

    last_yolo_run: bool = False

    motion_score: float = 0.0

    urgency_score: float = 0.0

    rider_count: int = 0

    avg_inference_ms: float = 0.0

    person_count: int = 0

    motorcycle_count: int = 0

    rider_pair_count: int = 0

    preprocess_ms: float = 0.0

    infer_ms: float = 0.0

    draw_ms: float = 0.0

    total_ms: float = 0.0

    def reset(
        self,
        scheduler,
        motion_estimator,
        classifier
    ):

        self.cached_detections.clear()

        self.violation_count = 0

        self.yolo_history.clear()

        scheduler.reset()

        motion_estimator.prev_gray = None

        classifier.reset()
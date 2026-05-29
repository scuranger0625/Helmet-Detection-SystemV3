from core.head_detector import HeadDetector
from core.motion import MotionEstimator
from core.scheduler import BOSScheduler
from core.detector import YOLODetector
from core.classifier import HeadRiskEstimator

from utils.metrics import FPSCounter
from utils.evaluator import Evaluator


class InferenceModules:
    def __init__(self):

        self.head_detector = HeadDetector()
        self.detector = YOLODetector()
        self.classifier = HeadRiskEstimator()
        self.scheduler = BOSScheduler(
            base_motion_threshold=0.03,
            urgency_threshold=0.40,
            force_refresh=5
        )

        self.motion_estimator = MotionEstimator()
        self.fps_counter = FPSCounter()
        self.evaluator = Evaluator()

    def reset(self, runtime):

        runtime.reset(
            self.scheduler,
            self.motion_estimator,
            self.classifier
        )
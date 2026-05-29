import numpy as np


class Evaluator:

    def __init__(self):

        self.total_frames = 0
        self.yolo_calls = 0
        self.violation_count = 0

        # time-series arrays for percentiles
        self.fps_values = []
        self.preprocess_ms_values = []
        self.infer_ms_values = []
        self.draw_ms_values = []
        self.total_ms_values = []

    def update(
        self,
        fps,
        yolo_run,
        frame_violations=0,
        preprocess_ms=0.0,
        infer_ms=0.0,
        draw_ms=0.0,
        total_ms=0.0
    ):

        self.total_frames += 1

        if yolo_run:
            self.yolo_calls += 1

        self.violation_count += int(frame_violations or 0)

        # append statistics
        try:
            self.fps_values.append(float(fps))
        except Exception:
            pass

        self.preprocess_ms_values.append(float(preprocess_ms or 0.0))
        self.infer_ms_values.append(float(infer_ms or 0.0))
        self.draw_ms_values.append(float(draw_ms or 0.0))
        self.total_ms_values.append(float(total_ms or 0.0))

    def _stats(self, arr):
        if not arr:
            return {
                "avg": 0.0,
                "min": 0.0,
                "p5": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }

        a = np.array(arr)
        return {
            "avg": float(np.mean(a)),
            "min": float(np.min(a)),
            "p5": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
        }

    def summary(self):

        fps_stats = self._stats(self.fps_values)
        preprocess_stats = self._stats(self.preprocess_ms_values)
        infer_stats = self._stats(self.infer_ms_values)
        draw_stats = self._stats(self.draw_ms_values)
        total_stats = self._stats(self.total_ms_values)

        yolo_ratio = float(self.yolo_calls) / max(self.total_frames, 1)
        saved_ratio = 1.0 - yolo_ratio

        return {
            "saved_ratio": saved_ratio * 100.0,
            "total_frames": self.total_frames,
            "avg_fps": fps_stats["avg"],
            "min_fps": fps_stats["min"],
            "p5_fps": fps_stats["p5"],
            "p50_fps": fps_stats["p50"],
            "p95_fps": fps_stats["p95"],
            "p99_fps": fps_stats["p99"],
            "preprocess_ms_avg": preprocess_stats["avg"],
            "infer_ms_avg": infer_stats["avg"],
            "draw_ms_avg": draw_stats["avg"],
            "total_ms_avg": total_stats["avg"],
            "yolo_calls": self.yolo_calls,
            "yolo_ratio": yolo_ratio,
            "violations": self.violation_count,
        }
import cv2
import numpy as np
from collections import deque


class HeadRiskEstimator:

    def __init__(
        self,
        low_risk_threshold=0.35,
        high_risk_threshold=0.65,
        history_size=10
    ):

        # ==========================================
        # 低風險門檻
        # risk_score < 0.35
        # 判定為 HELMET
        # ==========================================
        self.low_risk_threshold = low_risk_threshold

        # ==========================================
        # 高風險門檻
        # risk_score >= 0.65
        # 判定為 NO HELMET
        # ==========================================
        self.high_risk_threshold = high_risk_threshold

        # ==========================================
        # Temporal voting / smoothing
        # 保存最近 N 次 head risk
        # 用平均值降低單幀誤殺
        # ==========================================
        self.risk_history = deque(
            maxlen=history_size
        )

    def reset(
        self
    ):

        # ==========================================
        # seek / 換影片 / 重置狀態時可以呼叫
        # ==========================================
        self.risk_history.clear()

    def compute_skin_ratio(
        self,
        head_crop
    ):

        # ==========================================
        # BGR -> HSV
        # ==========================================
        hsv = cv2.cvtColor(
            head_crop,
            cv2.COLOR_BGR2HSV
        )

        # ==========================================
        # 粗略膚色範圍
        #
        # 注意：
        # 這不是精準膚色模型
        # 只是用來估計「裸頭風險」
        # ==========================================
        skin_mask = cv2.inRange(
            hsv,
            (0, 20, 60),
            (25, 180, 255)
        )

        skin_ratio = (
            skin_mask.mean() / 255.0
        )

        return float(
            np.clip(
                skin_ratio,
                0.0,
                1.0
            )
        )

    def compute_edge_density(
        self,
        head_crop
    ):

        # ==========================================
        # 頭髮通常 edge 較多
        # 安全帽表面通常比較平滑
        # ==========================================
        gray = cv2.cvtColor(
            head_crop,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            80,
            160
        )

        edge_density = (
            edges.mean() / 255.0
        )

        return float(
            np.clip(
                edge_density,
                0.0,
                1.0
            )
        )

    def compute_texture_variance(
        self,
        head_crop
    ):

        # ==========================================
        # texture 複雜度
        #
        # 使用 std 而不是 var
        # var 太容易暴衝
        # ==========================================
        gray = cv2.cvtColor(
            head_crop,
            cv2.COLOR_BGR2GRAY
        )

        texture_variance = (
            np.std(gray) / 128.0
        )

        return float(
            np.clip(
                texture_variance,
                0.0,
                1.0
            )
        )

    def compute_circle_confidence(
        self,
        head_crop
    ):

        # ==========================================
        # 安全帽常有半球 / 圓弧結構
        # 用 Hough Circle 做便宜幾何偵測
        # ==========================================
        gray = cv2.cvtColor(
            head_crop,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        h, w = gray.shape[:2]

        # ==========================================
        # crop 太小時不要硬做 Hough
        # 避免雜訊誤判
        # ==========================================
        if h < 20 or w < 20:
            return 0.0

        min_radius = max(
            5,
            int(min(h, w) * 0.15)
        )

        max_radius = max(
            min_radius + 1,
            int(min(h, w) * 0.60)
        )

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(10, min(h, w) // 2),
            param1=50,
            param2=20,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is None:
            return 0.0

        # ==========================================
        # 找到圓弧不代表一定是安全帽
        # 所以 confidence 不給太滿
        # ==========================================
        count = len(circles[0])

        confidence = min(
            count / 3.0,
            1.0
        )

        return float(
            np.clip(
                confidence,
                0.0,
                1.0
            )
        )

    def compute_reflective_ratio(
        self,
        head_crop
    ):

        # ==========================================
        # 安全帽塑膠表面常有高亮反光
        # 但光頭 / 白髮 / 白衣背景也可能誤判
        # 所以只當輔助特徵
        # ==========================================
        hsv = cv2.cvtColor(
            head_crop,
            cv2.COLOR_BGR2HSV
        )

        value = hsv[:, :, 2]

        bright = cv2.threshold(
            value,
            220,
            255,
            cv2.THRESH_BINARY
        )[1]

        reflective_ratio = (
            bright.mean() / 255.0
        )

        return float(
            np.clip(
                reflective_ratio,
                0.0,
                1.0
            )
        )

    def estimate(
        self,
        head_crop
    ):

        # ==========================================
        # 防呆：
        # 空 crop 直接回傳高不確定
        # ==========================================
        if head_crop is None or head_crop.size == 0:

            return {
                "possible_no_helmet": False,
                "status": "UNKNOWN",
                "risk_score": 0.5,
                "instant_risk": 0.5,
                "helmet_score": 0.5,
                "skin_ratio": 0.0,
                "edge_density": 0.0,
                "texture_variance": 0.0,
                "circle_confidence": 0.0,
                "reflective_ratio": 0.0,
                "uncertainty": 1.0
            }

        # ==========================================
        # 1. 特徵抽取
        # ==========================================
        skin_ratio = self.compute_skin_ratio(
            head_crop
        )

        edge_density = self.compute_edge_density(
            head_crop
        )

        texture_variance = self.compute_texture_variance(
            head_crop
        )

        circle_confidence = self.compute_circle_confidence(
            head_crop
        )

        reflective_ratio = self.compute_reflective_ratio(
            head_crop
        )

        # ==========================================
        # 2. Helmet Score
        #
        # 越高代表越像安全帽
        #
        # circle_confidence:
        #   圓弧結構
        #
        # reflective_ratio:
        #   反光塑膠表面
        #
        # 1 - edge_density:
        #   越平滑越像安全帽
        #
        # 1 - texture_variance:
        #   texture 越單純越像安全帽
        #
        # skin_ratio:
        #   膚色越高，越不像安全帽
        # ==========================================
        helmet_score = (
            0.30 * circle_confidence +
            0.20 * reflective_ratio +
            0.20 * (1.0 - edge_density) +
            0.20 * (1.0 - texture_variance) +
            0.10 * (1.0 - skin_ratio)
        )

        helmet_score = float(
            np.clip(
                helmet_score,
                0.0,
                1.0
            )
        )

        # ==========================================
        # 3. Instant Risk
        #
        # 單幀裸頭風險
        # 越高越可能沒戴安全帽
        # ==========================================
        instant_risk = (
            1.0 - helmet_score
        )

        instant_risk = float(
            np.clip(
                instant_risk,
                0.0,
                1.0
            )
        )

        # ==========================================
        # 4. Temporal Smoothing
        #
        # 不用單幀直接判斷
        # 用最近 N 次平均降低誤殺
        # ==========================================
        self.risk_history.append(
            instant_risk
        )

        risk_score = float(
            np.mean(
                self.risk_history
            )
        )

        risk_score = float(
            np.clip(
                risk_score,
                0.0,
                1.0
            )
        )

        # ==========================================
        # 5. 三段式狀態
        #
        # HELMET:
        #   低風險
        #
        # SUSPICIOUS:
        #   中間灰區，不硬判
        #
        # NO HELMET:
        #   高風險
        # ==========================================
        if risk_score < self.low_risk_threshold:

            status = "HELMET"

        elif risk_score < self.high_risk_threshold:

            status = "SUSPICIOUS"

        else:

            status = "NO HELMET"

        possible_no_helmet = (
            status == "NO HELMET"
        )

        # ==========================================
        # 6. Uncertainty
        #
        # 越靠近中間區間中心，越不確定
        # 越遠離灰區，越確定
        # ==========================================
        middle = (
            self.low_risk_threshold +
            self.high_risk_threshold
        ) / 2.0

        half_width = (
            self.high_risk_threshold -
            self.low_risk_threshold
        ) / 2.0

        uncertainty = (
            1.0 -
            min(
                abs(risk_score - middle) /
                max(half_width, 1e-6),
                1.0
            )
        )

        uncertainty = float(
            np.clip(
                uncertainty,
                0.0,
                1.0
            )
        )

        return {

            # ======================================
            # 給 main.py 用
            # ======================================
            "possible_no_helmet":
                possible_no_helmet,

            "status":
                status,

            "risk_score":
                risk_score,

            "instant_risk":
                instant_risk,

            "helmet_score":
                helmet_score,

            # ======================================
            # debug / report 用
            # ======================================
            "skin_ratio":
                skin_ratio,

            "edge_density":
                edge_density,

            "texture_variance":
                texture_variance,

            "circle_confidence":
                circle_confidence,

            "reflective_ratio":
                reflective_ratio,

            "uncertainty":
                uncertainty
        }


# ==========================================
# Backward compatibility
#
# 讓舊 main.py 如果還在：
# from core.classifier import HelmetClassifier
#
# 也不會直接爆炸。
# ==========================================
HelmetClassifier = HeadRiskEstimator
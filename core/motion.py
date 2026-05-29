# MGF-GCN like

import cv2
import numpy as np


class MotionEstimator:

    def __init__(

        self,

        grid_size=4

    ):

        # previous gray frame
        self.prev_gray = None

        # spatial grid
        self.grid_size = grid_size

    def compute_motion_map(

        self,

        frame

    ):

        # ==========================================
        # BGR -> Gray
        # ==========================================
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # 第一幀
        if self.prev_gray is None:

            self.prev_gray = gray

            return np.zeros(
                (
                    self.grid_size,
                    self.grid_size
                )
            )

        # ==========================================
        # frame difference
        # ==========================================
        diff = cv2.absdiff(
            self.prev_gray,
            gray
        )

        # ==========================================
        # threshold
        # ==========================================
        _, thresh = cv2.threshold(
            diff,
            25,
            255,
            cv2.THRESH_BINARY
        )

        h, w = thresh.shape

        cell_h = h // self.grid_size
        cell_w = w // self.grid_size

        motion_map = np.zeros(
            (
                self.grid_size,
                self.grid_size
            )
        )

        # ==========================================
        # compute motion per grid
        # ==========================================
        for gy in range(self.grid_size):

            for gx in range(self.grid_size):

                y1 = gy * cell_h
                y2 = (gy + 1) * cell_h

                x1 = gx * cell_w
                x2 = (gx + 1) * cell_w

                cell = thresh[
                    y1:y2,
                    x1:x2
                ]

                if cell.size == 0:
                    continue

                score = (
                    cell.mean() / 255.0
                )

                motion_map[gy, gx] = score

        # update previous frame
        self.prev_gray = gray

        return motion_map

    def compute_motion(

        self,

        frame

    ):

        motion_map = self.compute_motion_map(
            frame
        )

        # ==========================================
        # global motion
        # ==========================================
        motion_score = np.max(
            motion_map
        )

        return float(motion_score)
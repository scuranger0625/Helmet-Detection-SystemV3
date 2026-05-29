import time


class FPSCounter:

    def __init__(self):

        self.prev_time = time.time()
        self.fps = 0.0

    def update(self):

        current_time = time.time()

        fps = 1.0 / max(
            current_time - self.prev_time,
            1e-6
        )

        self.prev_time = current_time

        return fps
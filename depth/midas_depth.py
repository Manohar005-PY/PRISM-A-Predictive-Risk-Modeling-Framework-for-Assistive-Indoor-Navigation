import torch
import cv2
import numpy as np

class DepthEstimator:

    def __init__(self):
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to("cuda")
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.small_transform

        self.depth_map = None
        self.depth_norm = None
        self.counter = 0

    def compute(self, frame):
        self.counter += 1

        # run every 2 frames
        if self.counter % 2 != 0:
            return self.depth_norm

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to("cuda")

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # ---- Normalize ONCE per frame ----
        d_min = depth.min()
        d_max = depth.max()

        self.depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)

        return self.depth_norm

    def get_distance(self, bbox):
        if self.depth_norm is None:
            return 0

        x1, y1, x2, y2 = bbox
        roi = self.depth_norm[y1:y2, x1:x2]

        if roi.size == 0:
            return 0

        D = float(np.mean(roi))

        return D
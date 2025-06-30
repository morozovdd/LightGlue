import kornia
import torch

from .utils import Extractor


class DISK(Extractor):
    default_conf = {
        "weights": "depth",
        "max_num_keypoints": None,
        "desc_dim": 128,
        "nms_window_size": 5,
        "detection_threshold": 0.0,
        "pad_if_not_divisible": True,
    }

    preprocess_conf = {
        "resize": 1024,
        "grayscale": False,
    }

    required_data_keys = ["image"]

    def __init__(self, **conf) -> None:
        super().__init__(**conf)  # Update with default configuration.
        self.model = kornia.feature.DISK.from_pretrained(self.conf.weights)

    def forward(self, data: dict) -> dict:
        """Compute keypoints, scores, descriptors for image"""
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        image = data["image"]
        if image.shape[1] == 1:
            image = kornia.color.grayscale_to_rgb(image)
        features = self.model(
            image,
            n=self.conf.max_num_keypoints,
            window_size=self.conf.nms_window_size,
            score_threshold=self.conf.detection_threshold,
            pad_if_not_divisible=self.conf.pad_if_not_divisible,
        )
        keypoints = [f.keypoints for f in features]
        scores = [f.detection_scores for f in features]
        descriptors = [f.descriptors for f in features]
        del features

        keypoints = torch.stack(keypoints, 0)
        scores = torch.stack(scores, 0)
        descriptors = torch.stack(descriptors, 0)

        return {
            "keypoints": keypoints.to(image).contiguous(),
            "keypoint_scores": scores.to(image).contiguous(),
            "descriptors": descriptors.to(image).contiguous(),
        }

    def _compute_dense_descriptors_fallback(self, img: torch.Tensor) -> torch.Tensor:
        """Fallback method for DISK to extract dense descriptors

        Note: DISK uses Kornia's implementation which doesn't directly expose dense descriptors.
        This is a limitation of the current Kornia DISK implementation.

        Args:
            img: Input image tensor [B, C, H, W]

        Returns:
            Dense descriptor tensor [B, C, H', W']
        """
        raise NotImplementedError(
            "DISK model doesn't support custom keypoint extraction due to limitations "
            "in the Kornia implementation. The Kornia DISK model doesn't expose dense "
            "descriptor maps, only keypoint-based descriptors. "
            "\n\nSupported models for custom keypoint extraction:"
            "\n- SuperPoint"
            "\n- ALIKED"
            "\n\nIf you need DISK with custom keypoints, you may need to use a different "
            "DISK implementation that exposes dense feature maps."
        )

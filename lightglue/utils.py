import collections.abc as collections
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union

import cv2
import kornia
import numpy as np
import torch


class ImagePreprocessor:
    default_conf = {
        "resize": None,  # target edge length, None for no resizing
        "side": "long",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
    }

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**self.conf)

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        if self.conf.resize is not None:
            img = kornia.geometry.transform.resize(
                img,
                self.conf.resize,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
            )
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        return img, scale


def map_tensor(input_, func: Callable):
    string_classes = (str, bytes)
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif isinstance(input_, torch.Tensor):
        return func(input_)
    else:
        return input_


def batch_to_device(batch: dict, device: str = "cpu", non_blocking: bool = True):
    """Move batch (dict) to device"""

    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking).detach()

    return map_tensor(batch, _func)


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str = "max",
    interp: Optional[str] = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def load_image(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    image = read_image(path)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)


class Extractor(torch.nn.Module):
    def __init__(self, **conf):
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})

    @torch.no_grad()
    def extract(self, img: torch.Tensor, **conf) -> dict:
        """Perform extraction with online resizing"""
        if img.dim() == 3:
            img = img[None]  # add batch dim
        assert img.dim() == 4 and img.shape[0] == 1
        shape = img.shape[-2:][::-1]
        img, scales = ImagePreprocessor(**{**self.preprocess_conf, **conf})(img)
        feats = self.forward({"image": img})
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5
        return feats

    @torch.no_grad()
    def extract_descriptors_at_keypoints(
        self, 
        img: torch.Tensor, 
        keypoints: torch.Tensor, 
        **conf
    ) -> dict:
        """Extract descriptors at custom keypoint locations
        
        Args:
            img: Input image tensor [H, W] or [C, H, W] or [B, C, H, W]
            keypoints: Keypoint coordinates [N, 2] or [B, N, 2] in (x, y) format
            **conf: Additional preprocessing configuration
            
        Returns:
            dict containing:
                - descriptors: [B, N, D] tensor of descriptors
                - keypoints: [B, N, 2] tensor of keypoints in image coordinates
                - image_size: [B, 2] tensor of original image size
        """
        # Handle input dimensions
        if img.dim() == 2:
            img = img[None, None]  # H,W -> B,C,H,W
        elif img.dim() == 3:
            img = img[None]  # C,H,W -> B,C,H,W
        assert img.dim() == 4, f"Image must be 2D, 3D or 4D, got {img.dim()}D"
        
        # Handle keypoints dimensions
        if keypoints.dim() == 2:
            keypoints = keypoints[None]  # N,2 -> B,N,2
        assert keypoints.dim() == 3 and keypoints.shape[-1] == 2, \
            f"Keypoints must be [N,2] or [B,N,2], got {keypoints.shape}"
        
        batch_size = img.shape[0]
        assert keypoints.shape[0] == batch_size or keypoints.shape[0] == 1, \
            f"Batch size mismatch: img={batch_size}, keypoints={keypoints.shape[0]}"
        
        # Expand keypoints to match batch size if needed
        if keypoints.shape[0] == 1 and batch_size > 1:
            keypoints = keypoints.expand(batch_size, -1, -1)
        
        original_shape = img.shape[-2:][::-1]  # (W, H)
        
        # Preprocess image (resize if needed)
        img, scales = ImagePreprocessor(**{**self.preprocess_conf, **conf})(img)
        
        # Scale keypoints to match resized image
        scaled_keypoints = (keypoints + 0.5) * scales[None] - 0.5
        
        # Forward pass to get feature maps (without keypoint detection)
        # We need to call the underlying network to get dense descriptors
        if hasattr(self, '_extract_dense_descriptors'):
            # If the model provides a method to extract dense descriptors
            dense_descriptors = self._extract_dense_descriptors({"image": img})
        elif hasattr(self, 'extract_dense_map'):
            # For ALIKED-style models
            feature_map, _ = self.extract_dense_map(img)
            dense_descriptors = feature_map
        else:
            # Fallback: run full forward pass and try to extract dense descriptors
            try:
                dense_descriptors = self._compute_dense_descriptors_fallback(img)
            except NotImplementedError:
                raise NotImplementedError(
                    f"Extractor {self.__class__.__name__} doesn't support dense descriptor extraction. "
                    "Please implement one of: '_extract_dense_descriptors', 'extract_dense_map', "
                    "or '_compute_dense_descriptors_fallback' methods."
                )
        
        # Sample descriptors at keypoint locations
        descriptors = self._sample_descriptors_at_locations(
            scaled_keypoints, dense_descriptors
        )
        
        return {
            "descriptors": descriptors,
            "keypoints": keypoints,  # Return original keypoints
            "image_size": torch.tensor(original_shape)[None].to(img).float().expand(batch_size, -1),
        }
    
    def _sample_descriptors_at_locations(
        self, 
        keypoints: torch.Tensor, 
        descriptors: torch.Tensor
    ) -> torch.Tensor:
        """Sample descriptors at given keypoint locations using bilinear interpolation
        
        Args:
            keypoints: [B, N, 2] keypoint coordinates in (x, y) format
            descriptors: [B, C, H, W] dense descriptor map
            
        Returns:
            [B, N, C] sampled descriptors
        """
        b, c, h, w = descriptors.shape
        _, n, _ = keypoints.shape
        
        # Normalize keypoints to [-1, 1] for grid_sample
        normalized_kpts = keypoints.clone()
        normalized_kpts[..., 0] = 2.0 * keypoints[..., 0] / (w - 1) - 1.0  # x coordinate
        normalized_kpts[..., 1] = 2.0 * keypoints[..., 1] / (h - 1) - 1.0  # y coordinate
        
        # Reshape for grid_sample: [B, 1, N, 2]
        grid = normalized_kpts.unsqueeze(1)
        
        # Sample descriptors
        args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
        sampled = torch.nn.functional.grid_sample(
            descriptors, grid, mode="bilinear", padding_mode="border", **args
        )
        
        # Reshape to [B, C, N] then transpose to [B, N, C]
        sampled = sampled.squeeze(2).transpose(1, 2)
        
        # Normalize descriptors
        sampled = torch.nn.functional.normalize(sampled, p=2, dim=-1)
        
        return sampled
    
    def _compute_dense_descriptors_fallback(self, img: torch.Tensor) -> torch.Tensor:
        """Fallback method to compute dense descriptors for models that don't expose them directly
        
        This method attempts different strategies to extract dense descriptors from models
        that may not have explicit support for it.
        
        Args:
            img: Input image tensor [B, C, H, W]
            
        Returns:
            Dense descriptor tensor [B, C, H', W']
        """
        # Strategy 1: Try to extract from feature computation parts
        if hasattr(self, 'model') and hasattr(self.model, 'extract_features'):
            # For models like DISK that might have feature extraction
            try:
                features = self.model.extract_features(img)
                if hasattr(features, 'descriptors'):
                    return features.descriptors
            except:
                pass
        
        # Strategy 2: Run forward and look for intermediate outputs
        # This is a more aggressive fallback that may not work for all models
        if hasattr(self, 'model'):
            # For wrapped models, try to access internal feature computation
            try:
                # This is model-specific and may need customization
                raise NotImplementedError(
                    f"Dense descriptor extraction not implemented for {self.__class__.__name__}. "
                    "This model may not support custom keypoint extraction."
                )
            except:
                pass
        
        raise NotImplementedError(
            f"Cannot extract dense descriptors from {self.__class__.__name__}. "
            "This model doesn't support custom keypoint extraction with the current implementation."
        )


def match_pair(
    extractor,
    matcher,
    image0: torch.Tensor,
    image1: torch.Tensor,
    device: str = "cpu",
    **preprocess,
):
    """Match a pair of images (image0, image1) with an extractor and matcher"""
    feats0 = extractor.extract(image0, **preprocess)
    feats1 = extractor.extract(image1, **preprocess)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    data = [feats0, feats1, matches01]
    # remove batch dim and move to target device
    feats0, feats1, matches01 = [batch_to_device(rbd(x), device) for x in data]
    return feats0, feats1, matches01

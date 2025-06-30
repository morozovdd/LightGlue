# Custom Keypoint Extraction for LightGlue Extractors

This implementation adds the ability to extract descriptors at custom keypoint locations for LightGlue feature extractors.

## Supported Extractors

| Extractor | Support Level | Notes |
|-----------|---------------|-------|
| **SuperPoint** | ✅ Full Support | Native implementation with dense descriptor extraction |
| **ALIKED** | ✅ Full Support | Uses existing `extract_dense_map` method |
| **DISK** | ❌ Limited Support | Kornia implementation doesn't expose dense descriptors |

## Usage

### Basic Usage

```python
import torch
from lightglue import SuperPoint

# Initialize extractor
extractor = SuperPoint(max_num_keypoints=2048)

# Load your image (as torch tensor)
image = torch.randn(1, 480, 640)  # [C, H, W] or [B, C, H, W]

# Define custom keypoints (x, y coordinates)
custom_keypoints = torch.tensor([
    [100.0, 150.0],
    [200.0, 250.0],
    [300.0, 100.0],
])

# Extract descriptors at custom keypoints
with torch.no_grad():
    result = extractor.extract_descriptors_at_keypoints(
        image, 
        custom_keypoints,
        resize=1024  # Optional preprocessing
    )

# Access results
descriptors = result['descriptors']  # [B, N, D] - normalized descriptors
keypoints = result['keypoints']      # [B, N, 2] - original keypoint coordinates
image_size = result['image_size']    # [B, 2] - original image size (W, H)
```

### Batch Processing

```python
# Process multiple images with different keypoints
batch_images = torch.randn(2, 1, 480, 640)  # [B, C, H, W]
batch_keypoints = torch.randn(2, 5, 2)       # [B, N, 2]

result = extractor.extract_descriptors_at_keypoints(
    batch_images, 
    batch_keypoints
)
```

### Working with Different Extractors

```python
# SuperPoint
from lightglue import SuperPoint
extractor = SuperPoint()

# ALIKED  
from lightglue import ALIKED
extractor = ALIKED(model_name="aliked-n16")

# Both support the same interface
result = extractor.extract_descriptors_at_keypoints(image, keypoints)
```

## Implementation Details

### How It Works

1. **Image Preprocessing**: Images are resized and preprocessed according to each extractor's requirements
2. **Dense Descriptor Extraction**: The implementation extracts dense descriptor maps from the network
3. **Keypoint Scaling**: Custom keypoints are scaled to match the processed image dimensions
4. **Bilinear Interpolation**: Descriptors are sampled at keypoint locations using bilinear interpolation
5. **Normalization**: Output descriptors are L2-normalized

### Architecture-Specific Methods

Each extractor implements the functionality differently:

- **SuperPoint**: Added `_extract_dense_descriptors()` method that runs the encoder and descriptor head
- **ALIKED**: Uses existing `extract_dense_map()` method and adds custom sampling
- **DISK**: Not supported due to Kornia model limitations

### Input/Output Specifications

**Inputs:**
- `img`: Image tensor with shapes `[H, W]`, `[C, H, W]`, or `[B, C, H, W]`
- `keypoints`: Keypoint coordinates with shapes `[N, 2]` or `[B, N, 2]` in (x, y) format
- `**conf`: Optional preprocessing configuration (e.g., `resize=1024`)

**Outputs:**
- `descriptors`: `[B, N, D]` - L2-normalized descriptors at keypoint locations
- `keypoints`: `[B, N, 2]` - Original keypoint coordinates (unchanged)
- `image_size`: `[B, 2]` - Original image size as (width, height)

## Error Handling

The implementation includes comprehensive error handling:

- **Dimension validation**: Ensures correct input tensor dimensions
- **Batch size matching**: Validates that image and keypoint batch sizes are compatible
- **Model compatibility**: Clear error messages for unsupported extractors
- **Graceful fallbacks**: Attempts multiple strategies for dense descriptor extraction

## Limitations

1. **DISK Support**: Limited due to Kornia's DISK implementation not exposing dense feature maps
2. **Memory Usage**: Dense descriptor extraction requires more memory than keypoint-only extraction
3. **Performance**: Slightly slower than standard extraction due to additional interpolation step

## Examples

See `example_custom_keypoints.py` for comprehensive usage examples with different extractors.

## Extending to Other Extractors

To add support for new extractors, implement one of these methods in your extractor class:

```python
def _extract_dense_descriptors(self, data: dict) -> torch.Tensor:
    """Extract dense descriptor map without keypoint detection"""
    # Return [B, C, H, W] dense descriptor tensor
    pass

# OR

def extract_dense_map(self, image: torch.Tensor) -> tuple:
    """Extract dense feature and score maps (ALIKED-style)"""
    # Return (feature_map, score_map) tuple
    pass

# OR

def _compute_dense_descriptors_fallback(self, img: torch.Tensor) -> torch.Tensor:
    """Custom fallback for dense descriptor extraction"""
    # Return [B, C, H, W] dense descriptor tensor
    pass
```

The base implementation will automatically detect and use these methods.

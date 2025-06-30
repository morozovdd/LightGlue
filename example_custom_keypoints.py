"""
Example script demonstrating how to extract descriptors at custom keypoints
using the enhanced Extractor class with different feature extractors.
"""

import torch
import numpy as np
from pathlib import Path

def test_extractor(extractor_class, extractor_name, **kwargs):
    """Test custom keypoint extraction with a specific extractor"""
    print(f"\n{'='*50}")
    print(f"Testing {extractor_name}")
    print(f"{'='*50}")
    
    try:
        # Initialize the extractor
        extractor = extractor_class(**kwargs)
        print(f"✅ Successfully initialized {extractor_name}")
        
        # Create a dummy image for demonstration
        if extractor_name == "DISK":
            image = torch.randn(3, 480, 640)  # RGB for DISK
        else:
            image = torch.randn(1, 480, 640)  # Grayscale for others
        print(f"Created dummy image with shape {image.shape}")
        
        # Define custom keypoints (in x, y format)
        custom_keypoints = torch.tensor([
            [100.0, 150.0],  # (x, y) coordinates
            [200.0, 250.0],
            [300.0, 100.0],
            [400.0, 200.0],
            [150.0, 300.0],
        ])
        
        print(f"Custom keypoints shape: {custom_keypoints.shape}")
        
        # Try to extract descriptors at the custom keypoints
        with torch.no_grad():
            try:
                result = extractor.extract_descriptors_at_keypoints(
                    image, 
                    custom_keypoints,
                    resize=1024
                )
                
                # Print results
                print(f"✅ Successfully extracted descriptors!")
                print(f"   Descriptors shape: {result['descriptors'].shape}")
                print(f"   Keypoints shape: {result['keypoints'].shape}")
                print(f"   Image size: {result['image_size']}")
                
                # Check descriptor normalization
                descriptors = result['descriptors']
                norms = torch.norm(descriptors[0], dim=1)
                print(f"   Descriptor norms (should be ~1.0): {norms.mean():.3f} ± {norms.std():.3f}")
                
                return True
                
            except NotImplementedError as e:
                print(f"❌ {extractor_name} doesn't support custom keypoint extraction:")
                print(f"   {str(e)}")
                return False
            except Exception as e:
                print(f"❌ Error during extraction with {extractor_name}:")
                print(f"   {str(e)}")
                return False
                
    except Exception as e:
        print(f"❌ Failed to initialize {extractor_name}:")
        print(f"   {str(e)}")
        return False

def main():
    """Test custom keypoint extraction with different extractors"""
    print("Testing Custom Keypoint Extraction with Different Feature Extractors")
    print("Note: This demo uses dummy data and may show import errors if dependencies aren't installed")
    
    # Test cases for different extractors
    test_cases = [
        # (module_path, class_name, display_name, init_kwargs)
        ("lightglue", "SuperPoint", "SuperPoint", {"max_num_keypoints": 2048}),
        ("lightglue", "ALIKED", "ALIKED", {"model_name": "aliked-n16"}),
        ("lightglue", "DISK", "DISK", {"max_num_keypoints": 2048}),
    ]
    
    results = {}
    
    for module_path, class_name, display_name, init_kwargs in test_cases:
        try:
            # Try to import the extractor class
            if module_path == "lightglue":
                try:
                    if class_name == "SuperPoint":
                        from lightglue import SuperPoint as ExtractorClass
                    elif class_name == "ALIKED":
                        from lightglue import ALIKED as ExtractorClass
                    elif class_name == "DISK":
                        from lightglue import DISK as ExtractorClass
                except ImportError as e:
                    print(f"\n❌ Could not import {class_name}: {e}")
                    results[display_name] = False
                    continue
            
            # Test the extractor
            success = test_extractor(ExtractorClass, display_name, **init_kwargs)
            results[display_name] = success
            
        except Exception as e:
            print(f"\n❌ Failed to test {display_name}: {e}")
            results[display_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Custom Keypoint Extraction Support")
    print(f"{'='*60}")
    
    for extractor_name, supported in results.items():
        status = "✅ SUPPORTED" if supported else "❌ NOT SUPPORTED"
        print(f"{extractor_name:<15} | {status}")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(f"{'='*60}")
    print("✅ SuperPoint: Full support - recommended for custom keypoint extraction")
    print("✅ ALIKED: Full support - good alternative with different architecture")
    print("❌ DISK: Limited support due to Kornia implementation constraints")
    print("\nFor best results with custom keypoints, use SuperPoint or ALIKED.")

if __name__ == "__main__":
    main()

"""
Script to test device availability (CUDA, MPS, or CPU)

Authors:
    Jéssica A. L. de Macêdo (jalm2@cin.ufpe.br)
    Matheus Borges Figueirôa (mbf3@cin.ufpe.br)
"""

import torch
import sys


def test_devices():
    """Test and display available compute devices"""

    print("=" * 80)
    print("Device Availability Test")
    print("=" * 80)

    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version}")

    # Test CUDA (NVIDIA GPUs)
    print("\n" + "-" * 80)
    print("CUDA (NVIDIA GPU)")
    print("-" * 80)
    if torch.cuda.is_available():
        print("✓ CUDA is available")
        print(f"  Number of devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(
                f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

        # Test CUDA operations
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            print("  ✓ CUDA operations test passed")
        except Exception as e:
            print(f"  ✗ CUDA operations test failed: {e}")
    else:
        print("✗ CUDA is not available")

    # Test MPS (Apple Silicon)
    print("\n" + "-" * 80)
    print("MPS (Apple Silicon GPU)")
    print("-" * 80)
    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            print("✓ MPS is available")
            if torch.backends.mps.is_built():
                print("  ✓ MPS is built")

                # Test MPS operations
                try:
                    x = torch.randn(100, 100).to('mps')
                    y = torch.randn(100, 100).to('mps')
                    z = torch.matmul(x, y)
                    print("  ✓ MPS operations test passed")
                except Exception as e:
                    print(f"  ✗ MPS operations test failed: {e}")
            else:
                print("  ✗ MPS is not built")
        else:
            print("✗ MPS is not available")
    else:
        print("✗ MPS backend not found (PyTorch version may be too old)")

    # CPU (always available)
    print("\n" + "-" * 80)
    print("CPU")
    print("-" * 80)
    print("✓ CPU is always available")
    try:
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.matmul(x, y)
        print("  ✓ CPU operations test passed")
    except Exception as e:
        print(f"  ✗ CPU operations test failed: {e}")

    # Recommendation
    print("\n" + "=" * 80)
    print("Recommendation")
    print("=" * 80)

    if torch.cuda.is_available():
        print("✓ Use CUDA for training (fastest option)")
        print(f"  Recommended device: cuda:0")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("✓ Use MPS for training (Apple Silicon acceleration)")
        print(f"  Recommended device: mps")
    else:
        print("⚠ Use CPU for training (slowest option)")
        print(f"  Recommended device: cpu")
        print("\nConsider using a GPU for faster training:")
        print("  - Cloud services: Google Colab, AWS, Azure, etc.")
        print("  - Local GPU: NVIDIA GPU with CUDA support")

    print("=" * 80)


if __name__ == '__main__':
    test_devices()

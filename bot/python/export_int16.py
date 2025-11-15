import torch
import numpy as np

from config import (
    FEATURE_DIM,
    HIDDEN1,
    HIDDEN2,
    HIDDEN3,
    HIDDEN4,
    HIDDEN5,
    RELU_CLIP,
    OUTPUT_SCALE_BITS,
    SCALE1,
    SCALE2,
)
from model import SimpleNNUE


def export_quantized(pt_path: str, bin_path: str):
    """
    Export a trained PyTorch SimpleNNUE model to quantized int16 binary format.
    
    Note: This exports the base layers only (not residual blocks) for simpler C++ implementation.
    For full model export including residual blocks, use ONNX or custom format.
    
    Args:
        pt_path: Path to the .pt file (e.g., 'nnue_state_dict.pt' or checkpoint file)
        bin_path: Output path for the binary file (e.g., 'nnue_weights.bin')
    """
    import os
    
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Model file not found: {pt_path}")
    
    print(f"Loading model from {pt_path}...")
    model = SimpleNNUE()
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    
    # Handle both checkpoint format and direct state_dict format
    if isinstance(state, dict) and 'model_state_dict' in state:
        print("Detected checkpoint format, extracting model_state_dict...")
        model.load_state_dict(state['model_state_dict'])
        if 'epoch' in state:
            print(f"  Checkpoint from epoch {state['epoch'] + 1}")
        if 'val_loss' in state:
            print(f"  Validation loss: {state['val_loss']:.4f}")
    else:
        model.load_state_dict(state)
    
    model.eval()
    print("Model loaded successfully!")
    print("\nWARNING: This export only includes base layers (fc1-fc5, fc_out).")
    print("Residual blocks, LayerNorm, and Dropout are NOT exported.")
    print("For production use, consider ONNX export or implement full architecture in C++.\n")

    # Extract base layer weights only (simplified export)
    print("Extracting base layer weights from model...")
    fc1_w = model.fc1.weight.detach().numpy()      # (HIDDEN1, FEATURE_DIM)
    fc1_b = model.fc1.bias.detach().numpy()        # (HIDDEN1,)
    
    fc2_w = model.fc2.weight.detach().numpy()      # (HIDDEN2, HIDDEN1)
    fc2_b = model.fc2.bias.detach().numpy()        # (HIDDEN2,)
    
    fc3_w = model.fc3.weight.detach().numpy()      # (HIDDEN3, HIDDEN2)
    fc3_b = model.fc3.bias.detach().numpy()        # (HIDDEN3,)
    
    fc4_w = model.fc4.weight.detach().numpy()      # (HIDDEN4, HIDDEN3)
    fc4_b = model.fc4.bias.detach().numpy()        # (HIDDEN4,)
    
    fc5_w = model.fc5.weight.detach().numpy()      # (HIDDEN5, HIDDEN4)
    fc5_b = model.fc5.bias.detach().numpy()        # (HIDDEN5,)
    
    out_w = model.fc_out.weight.detach().numpy()   # (1, HIDDEN5)
    out_b = model.fc_out.bias.detach().numpy()[0]  # scalar
    
    print(f"FC1 weights shape: {fc1_w.shape}")
    print(f"FC2 weights shape: {fc2_w.shape}")
    print(f"FC3 weights shape: {fc3_w.shape}")
    print(f"FC4 weights shape: {fc4_w.shape}")
    print(f"FC5 weights shape: {fc5_w.shape}")
    print(f"Output weights shape: {out_w.shape}")

    # Quantize weights (simple scaling, no SCALE1/SCALE2 for base export)
    print("\nQuantizing weights to int16...")
    w_fc1_q = np.round(fc1_w * SCALE1).astype(np.int16)
    b_fc1_q = np.round(fc1_b * SCALE1).astype(np.int32)

    w_fc2_q = np.round(fc2_w * SCALE1).astype(np.int16)
    b_fc2_q = np.round(fc2_b * SCALE1).astype(np.int32)

    w_fc3_q = np.round(fc3_w * SCALE1).astype(np.int16)
    b_fc3_q = np.round(fc3_b * SCALE1).astype(np.int32)

    w_fc4_q = np.round(fc4_w * SCALE1).astype(np.int16)
    b_fc4_q = np.round(fc4_b * SCALE1).astype(np.int32)

    w_fc5_q = np.round(fc5_w * SCALE1).astype(np.int16)
    b_fc5_q = np.round(fc5_b * SCALE1).astype(np.int32)

    w_out_q = np.round(out_w * SCALE1).astype(np.int16)
    b_out_q = np.round(out_b * SCALE1).astype(np.int32)

    print("Quantization ranges:")
    print(f"  fc1 weights:    [{w_fc1_q.min()}, {w_fc1_q.max()}]")
    print(f"  fc2 weights:    [{w_fc2_q.min()}, {w_fc2_q.max()}]")
    print(f"  fc3 weights:    [{w_fc3_q.min()}, {w_fc3_q.max()}]")
    print(f"  fc4 weights:    [{w_fc4_q.min()}, {w_fc4_q.max()}]")
    print(f"  fc5 weights:    [{w_fc5_q.min()}, {w_fc5_q.max()}]")
    print(f"  output weights: [{w_out_q.min()}, {w_out_q.max()}]")

    print(f"\nWriting binary file to {bin_path}...")
    with open(bin_path, "wb") as f:
        # Header: architecture dimensions
        f.write(np.int32(FEATURE_DIM).tobytes())
        f.write(np.int32(HIDDEN1).tobytes())
        f.write(np.int32(HIDDEN2).tobytes())
        f.write(np.int32(HIDDEN3).tobytes())
        f.write(np.int32(HIDDEN4).tobytes())
        f.write(np.int32(HIDDEN5).tobytes())

        # FC1: FEATURE_DIM -> HIDDEN1
        f.write(b_fc1_q.astype(np.int32).tobytes())
        f.write(w_fc1_q.astype(np.int16).tobytes())

        # FC2: HIDDEN1 -> HIDDEN2
        f.write(b_fc2_q.astype(np.int32).tobytes())
        f.write(w_fc2_q.astype(np.int16).tobytes())

        # FC3: HIDDEN2 -> HIDDEN3
        f.write(b_fc3_q.astype(np.int32).tobytes())
        f.write(w_fc3_q.astype(np.int16).tobytes())

        # FC4: HIDDEN3 -> HIDDEN4
        f.write(b_fc4_q.astype(np.int32).tobytes())
        f.write(w_fc4_q.astype(np.int16).tobytes())

        # FC5: HIDDEN4 -> HIDDEN5
        f.write(b_fc5_q.astype(np.int32).tobytes())
        f.write(w_fc5_q.astype(np.int16).tobytes())

        # Output: HIDDEN5 -> 1
        f.write(b_out_q.astype(np.int32).tobytes())
        f.write(w_out_q.astype(np.int16).tobytes())

    import os
    file_size = os.path.getsize(bin_path)
    print(f"\nSuccessfully exported quantized SimpleNNUE to {bin_path}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"\nBinary format:")
    print(f"  - FEATURE_DIM: {FEATURE_DIM}")
    print(f"  - HIDDEN1: {HIDDEN1}")
    print(f"  - HIDDEN2: {HIDDEN2}")
    print(f"  - HIDDEN3: {HIDDEN3}")
    print(f"  - HIDDEN4: {HIDDEN4}")
    print(f"  - HIDDEN5: {HIDDEN5}")
    print(f"  - Architecture: {FEATURE_DIM} -> {HIDDEN1} -> {HIDDEN2} -> {HIDDEN3} -> {HIDDEN4} -> {HIDDEN5} -> 1")
    print(f"  - Quantization scales: SCALE1={SCALE1}, SCALE2={SCALE2}")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export trained NNUE PyTorch model to quantized int16 binary format"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="nnue_state_dict.pt",
        help="Input PyTorch model file (.pt) (default: nnue_state_dict.pt)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="nnue_weights.bin",
        help="Output binary file (.bin) (default: nnue_weights.bin)"
    )
    
    args = parser.parse_args()
    
    try:
        export_quantized(args.input, args.output)
    except Exception as e:
        print(f"\nError during export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


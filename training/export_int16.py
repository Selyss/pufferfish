from __future__ import annotations

import struct
import sys
from pathlib import Path

import torch
from torch import nn

# Try to find model.py/model_compact.py in multiple locations
script_dir = Path(__file__).parent
repo_root = script_dir.parent

possible_paths = [
    script_dir,  # training/
    repo_root / "src" / "bot",  # legacy src/bot/
]

model_found = False
for path in possible_paths:
    sys.path.insert(0, str(path))
    try:
        from model import ResidualBlock, SimpleNNUE
        from model_compact import CompactNNUE, CompactResidualBlock
        model_found = True
        print(f"Found model.py in: {path}")
        break
    except ImportError:
        continue

if not model_found:
    print("Error: Cannot find model.py/model_compact.py")
    print("Searched in:")
    for path in possible_paths:
        print(f"  - {path}")
    sys.exit(1)

# Layer type constants
LINEAR = 1
LAYERNORM = 2
RESIDUAL = 3
COMPACT_RESIDUAL = 4


def load_checkpoint(path: str) -> tuple:
    """Load model from checkpoint file."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("model_state") or checkpoint
        config = checkpoint.get("model_config", {})
    else:
        state_dict = checkpoint
        config = {}

    model_type = config.get("model_type", "simple")

    # Get model config
    input_dim = config.get("input_dim", 795)
    hidden_dims = config.get("hidden_dims", [2048, 2048, 1024, 512, 256])
    dropout = config.get("dropout", 0.05)
    residual_repeats = config.get("residual_repeats", [2, 2, 2, 2, 2])

    # Create model
    if model_type == "compact":
        model = CompactNNUE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            residual_repeats=residual_repeats,
        )
    else:
        model = SimpleNNUE(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            residual_repeats=residual_repeats,
        )

    # Load weights
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)

    model.eval()
    return model, input_dim, len(hidden_dims), model_type


def write_linear(f, layer: nn.Linear) -> tuple:
    """Write LINEAR layer: type_id, out, in, weight[out][in], bias[out]"""
    weight = layer.weight.detach().cpu().float()
    bias = layer.bias.detach().cpu().float()

    out_features = weight.shape[0]
    in_features = weight.shape[1]

    f.write(struct.pack("<III", LINEAR, out_features, in_features))
    f.write(weight.numpy().tobytes())
    f.write(bias.numpy().tobytes())

    return out_features, in_features


def write_layernorm(f, layer: nn.LayerNorm) -> int:
    """Write LAYERNORM layer: type_id, size, 0, weight[size], bias[size], eps"""
    weight = layer.weight.detach().cpu().float()
    bias = layer.bias.detach().cpu().float()
    size = weight.shape[0]
    eps = layer.eps

    f.write(struct.pack("<III", LAYERNORM, size, 0))
    f.write(weight.numpy().tobytes())
    f.write(bias.numpy().tobytes())
    f.write(struct.pack("<f", eps))

    return size


def write_residual(f, block: ResidualBlock) -> int:
    """Write RESIDUAL block with LayerNorm."""
    w1 = block.lin1.weight.detach().cpu().float()
    b1 = block.lin1.bias.detach().cpu().float()

    w2 = block.lin2.weight.detach().cpu().float()
    b2 = block.lin2.bias.detach().cpu().float()

    norm_w = block.norm.weight.detach().cpu().float()
    norm_b = block.norm.bias.detach().cpu().float()
    eps = block.norm.eps

    dim = w1.shape[0]

    f.write(struct.pack("<III", RESIDUAL, dim, dim))
    f.write(w1.numpy().tobytes())
    f.write(b1.numpy().tobytes())
    f.write(w2.numpy().tobytes())
    f.write(b2.numpy().tobytes())
    f.write(norm_w.numpy().tobytes())
    f.write(norm_b.numpy().tobytes())
    f.write(struct.pack("<f", eps))

    return dim


def write_compact_residual(f, block: CompactResidualBlock) -> int:
    """Write COMPACT_RESIDUAL block: type_id, dim, dim, lin1, lin2 (no norm)."""
    w1 = block.lin1.weight.detach().cpu().float()
    b1 = block.lin1.bias.detach().cpu().float()
    w2 = block.lin2.weight.detach().cpu().float()
    b2 = block.lin2.bias.detach().cpu().float()

    dim = w1.shape[0]

    f.write(struct.pack("<III", COMPACT_RESIDUAL, dim, dim))
    f.write(w1.numpy().tobytes())
    f.write(b1.numpy().tobytes())
    f.write(w2.numpy().tobytes())
    f.write(b2.numpy().tobytes())

    return dim


def export_binary(checkpoint_path: str, output_path: str) -> None:
    """Export PyTorch checkpoint to binary format for C++."""
    print(f"Loading checkpoint: {checkpoint_path}")
    model, input_dim, num_stages, model_type = load_checkpoint(checkpoint_path)

    # Count layers
    layer_count = 0
    for module in model.backbone:
        if isinstance(module, (nn.Linear, nn.LayerNorm, ResidualBlock, CompactResidualBlock)):
            layer_count += 1
    layer_count += 1  # output head

    print("Model architecture:")
    print(f"  Model type: {model_type}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Stages: {num_stages}")
    print(f"  Total layers to export: {layer_count}")

    metadata = (
        f'{{"format":"residual-nnue-v1","input_dim":{input_dim},'
        f'"layer_count":{layer_count},"model_type":"{model_type}"}}'
    )
    metadata_bytes = metadata.encode("utf-8")

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", len(metadata_bytes)))
        f.write(metadata_bytes)
        f.write(struct.pack("<I", layer_count))

        exported = 0
        for module in model.backbone:
            if isinstance(module, nn.Linear):
                out_f, in_f = write_linear(f, module)
                print(f"  [{exported+1}/{layer_count}] LINEAR {in_f} -> {out_f}")
                exported += 1
            elif isinstance(module, nn.LayerNorm):
                size = write_layernorm(f, module)
                print(f"  [{exported+1}/{layer_count}] LAYERNORM {size}")
                exported += 1
            elif isinstance(module, ResidualBlock):
                dim = write_residual(f, module)
                print(f"  [{exported+1}/{layer_count}] RESIDUAL {dim}x{dim}")
                exported += 1
            elif isinstance(module, CompactResidualBlock):
                dim = write_compact_residual(f, module)
                print(f"  [{exported+1}/{layer_count}] COMPACT_RESIDUAL {dim}x{dim}")
                exported += 1

        out_f, in_f = write_linear(f, model.output_head)
        print(f"  [{exported+1}/{layer_count}] LINEAR {in_f} -> {out_f} (output head)")
        exported += 1

    file_size = Path(output_path).stat().st_size
    print(f"\n✓ Successfully exported {exported} layers")
    print(f"✓ Output: {output_path}")
    print(f"✓ File size: {file_size / 1024 / 1024:.2f} MB")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m training.export_int16 <checkpoint.pt> [output.bin]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "nnue_residual.bin"

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    try:
        export_binary(checkpoint_path, output_path)
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


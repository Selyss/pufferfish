# Training Export Utilities

This folder contains the minimal Python code needed to export a trained
PyTorch NNUE checkpoint (`.pt`) into the binary format used by the C++
engine (residual-nnue-v1 format).

Files:
- `export_int16.py` - converts `.pt` to `.bin` (supports simple/compact models)
- `model.py` - SimpleNNUE definition (matches training)
- `model_compact.py` - CompactNNUE definition (matches training)

## Export example

From the repo root:

```powershell
python -m training.export_int16 models\inference006.pt models\nnue_residual.bin
```

The exporter reads `model_config` from the checkpoint to determine
`model_type`, `input_dim`, and layer sizes.


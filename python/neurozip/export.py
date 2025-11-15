import torch
import struct

def export_tiny_lstm(checkpoint_path: str, output_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    hidden_size = int(ckpt["hidden_size"])
    state = ckpt["model_state"]

    # Extract weights
    W_ih = state["lstm.weight_ih_l0"]     # (4H, 256)
    W_hh = state["lstm.weight_hh_l0"]     # (4H, H)
    b_ih = state["lstm.bias_ih_l0"]       # (4H)
    b_hh = state["lstm.bias_hh_l0"]       # (4H)
    W_out = state["fc.weight"]            # (256, H)
    b_out = state["fc.bias"]              # (256)

    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", 256))         # inputSize
        f.write(struct.pack("<I", hidden_size)) # hiddenSize
        f.write(struct.pack("<I", 1))           # numLayers
        f.write(struct.pack("<I", 0))           # reserved

        # Write weight arrays as float32 little-endian
        for tensor in [W_ih, W_hh, b_ih, b_hh, W_out, b_out]:
            arr = tensor.contiguous().view(-1).cpu().numpy()
            f.write(arr.astype("float32").tobytes())

    print(f"[+] Exported to {output_path}")

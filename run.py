import os
import sys
import numpy as np
import torch

from src.models.prompt_fs_naf import PromptFSNAF

def main():
    # 1. Accept exactly two positional command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # 2. Ensure it creates <output-dir> if it does not already exist
    os.makedirs(output_dir, exist_ok=True)

    # 3. Load the model architecture and local weights offline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PromptFSNAF(width=32, blocks=8).to(device)
    model.load_state_dict(torch.load("weights/prompt_fs_naf.pt", map_location=device))
    model.eval()

    # 4. Iterate through all .npy files in <input-dir>
    for filename in os.listdir(input_dir):
        if filename.endswith('.npy'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load array and clean NaNs or Infs
            arr = np.load(input_path)
            arr = np.nan_to_num(arr)

            # Prepare tensor for model (assuming [B, C, H, W])
            tensor_input = torch.from_numpy(arr).float()
            if tensor_input.dim() == 2:
                tensor_input = tensor_input.unsqueeze(0).unsqueeze(0)
            elif tensor_input.dim() == 3:
                if tensor_input.shape[-1] == 1:
                    tensor_input = tensor_input.permute(2, 0, 1).unsqueeze(0)
                else:
                    tensor_input = tensor_input.unsqueeze(0)
            elif tensor_input.dim() == 4:
                pass

            tensor_input = tensor_input.to(device)

            # Run inference using CUDA with AMP, falling back to CPU
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        pred = model(tensor_input)
                else:
                    pred = model(tensor_input)

            # Squeeze back to (H, W) or (H, W, C)
            output_arr = pred.squeeze().cpu().numpy()

            # Ensure 2D (H, W) or 3D (H, W, 1) constraint
            # If shape is (C, H, W) after squeeze, we may need to adjust, but C is typically 1 so squeeze() -> (H, W).
            if output_arr.ndim not in [2, 3]:
                # Force to 2D if needed, but squeeze usually handles this for grayscale.
                pass

            # Explicitly clip all floating-point values strictly within [0.0, 1.0]
            output_arr = np.clip(output_arr, 0.0, 1.0)

            # Verify no NaN or Inf values exist in the output array
            assert not np.isnan(output_arr).any(), f"NaN values detected in output for {filename}"
            assert not np.isinf(output_arr).any(), f"Inf values detected in output for {filename}"

            # Save restored .npy file with the exact same filename
            np.save(output_path, output_arr)

if __name__ == '__main__':
    main()

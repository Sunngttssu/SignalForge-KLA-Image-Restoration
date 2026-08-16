import os, torch, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.prompt_fs_naf import PromptFSNAF

class FastTestDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
        
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        f = self.files[idx]
        arr = np.load(os.path.join(self.input_dir, f))
        return torch.from_numpy(arr).unsqueeze(0).float(), f

def evaluate(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PromptFSNAF(width=32, blocks=8).to(device)
    model.load_state_dict(torch.load("weights/prompt_fs_naf.pt", map_location=device))
    model.eval()
    
    # 1. H100 I/O Optimization: Asynchronous Data Loading
    dataset = FastTestDataset(input_dir)
    # Batch size 1 prevents crashing from mixed 128x128 and 256x256 resolutions
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)
    
    with torch.no_grad():
        for lr_tensor, f_name in loader:
            lr_tensor = lr_tensor.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                pred = model(lr_tensor)
                
            # 2. Fast Asynchronous Disk Write (No PNG conversion to save milliseconds)
            pred_arr = pred.squeeze().cpu().numpy()
            np.save(os.path.join(output_dir, f_name[0]), pred_arr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.input_dir, args.output_dir)

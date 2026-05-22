"""
Pipeline Debugger
Compares old PIL preprocessing vs new no-PIL preprocessing.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import torch.nn.functional as Fnn
import torchvision.transforms.functional as TF
from PIL import Image

def get_old_preprocess(frame, resolution, device, means, stds):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    img_tensor = TF.to_tensor(pil_img) # [0,1]
    img_gpu = img_tensor.to(device)
    img_gpu = TF.resize(img_gpu, [resolution, resolution])
    img_gpu = TF.normalize(img_gpu, means, stds)
    return img_gpu.unsqueeze(0)

def get_new_preprocess(frame, resolution, device, means, stds):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cpu_t = torch.from_numpy(rgb)
    gpu_u8 = cpu_t.to(device, non_blocking=True)
    gpu_f = gpu_u8.permute(2, 0, 1).float().mul_(1.0 / 255.0)
    
    mean_gpu = torch.tensor(means, device=device).view(3, 1, 1)
    std_gpu = torch.tensor(stds, device=device).view(3, 1, 1)
    
    gpu_f = Fnn.interpolate(
        gpu_f.unsqueeze(0),
        size=(resolution, resolution),
        mode='bilinear',
        align_corners=False,
        antialias=True,
    )
    
    gpu_f = (gpu_f - mean_gpu) / std_gpu
    
    return gpu_f

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resolution = 384
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    # Dummy frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    old_out = get_old_preprocess(frame, resolution, device, means, stds)
    new_out = get_new_preprocess(frame, resolution, device, means, stds)
    
    diff = torch.abs(old_out - new_out)
    print(f"Max diff: {diff.max().item():.6f}")
    print(f"Mean diff: {diff.mean().item():.6f}")
    
    print("Old shape:", old_out.shape)
    print("New shape:", new_out.shape)
    
    if diff.max().item() > 1e-4:
        print("Significant difference found!")
        print("Testing intermediate steps...")
        
        # Test just the HWC->CHW and /255
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        t_old = TF.to_tensor(pil_img).to(device)
        
        t_new = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float().mul_(1.0 / 255.0)
        
        print("Base tensor diff:", torch.abs(t_old - t_new).max().item())
        
        # Test resize
        t_old_res = TF.resize(t_old, [resolution, resolution])
        t_new_res = Fnn.interpolate(t_new.unsqueeze(0), size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze(0)
        print("Resize diff:", torch.abs(t_old_res - t_new_res).max().item())
        
        # What if we use align_corners=True?
        t_new_res2 = Fnn.interpolate(t_new.unsqueeze(0), size=(resolution, resolution), mode='bilinear', align_corners=True).squeeze(0)
        print("Resize (align_corners=True) diff:", torch.abs(t_old_res - t_new_res2).max().item())
        
        # What about antialias=True?
        t_new_res3 = Fnn.interpolate(t_new.unsqueeze(0), size=(resolution, resolution), mode='bilinear', antialias=True).squeeze(0)
        print("Resize (antialias=True) diff:", torch.abs(t_old_res - t_new_res3).max().item())

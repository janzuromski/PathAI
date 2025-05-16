import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download

def main():
    with open('hf_token.txt', 'r') as token_file:
        token = token_file.readline().strip()
    login(token=token)
    local_dir = "/exports/path-cutane-lymfomen-hpc/jan/.cache/huggingface/mahmood_uni"
    os.makedirs(local_dir, exist_ok=True)
    hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)

if __name__ == '__main__':
    main()

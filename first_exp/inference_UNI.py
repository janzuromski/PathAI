import os
from PIL import Image
import timm
import torch
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

def main():
    project_dir = '/exports/path-cutane-lymfomen-hpc/jan/rl'
    model_dir = '/exports/path-cutane-lymfomen-hpc/jan/.cache/huggingface/mahmood_uni'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device.')
    img_path = os.path.join(project_dir, 'test/data/LUMC_H04_01233_1A.tiff')
    img = Image.open(img_path)
    transform = Compose([Resize(224), ToTensor(), Normalize(0.5, 0.5)])
    img = transform(img).unsqueeze(0).to(device)
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, 
        num_classes=0, dynamic_img_size=True
    )
    state_dict = torch.load(os.path.join(model_dir, "pytorch_model.bin"))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        feature_embedding = model(img)
        torch.save(
            feature_embedding, 
            os.path.join(project_dir, 'first_exp/embedding.pth')
        )

if __name__ == '__main__':
    main()
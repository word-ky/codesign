import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
import random
class GaussianNoise:
    def __init__(self, std=0.1):
        self.std = std
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0, 1)

class SaltPepperNoise:
    def __init__(self, prob=0.05):
        self.prob = prob
    def __call__(self, tensor):
        noise_mask = torch.rand_like(tensor)
        salt = (noise_mask < self.prob / 2).float()
        pepper = ((noise_mask >= self.prob / 2) & (noise_mask < self.prob)).float()
        
        result = tensor.clone()
        result = result * (1 - salt - pepper) + salt
        return result

class PoissonNoise:
    def __init__(self, scale=1.0):
        self.scale = scale
    def __call__(self, tensor):

        tensor_scaled = tensor * 255.0 / self.scale
        noisy = torch.poisson(tensor_scaled) * self.scale / 255.0
        return torch.clamp(noisy, 0, 1)

class JPEGCompression:
    def __init__(self, quality=30):
        self.quality = quality
    def __call__(self, img):
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)

class MotionBlur:
    def __init__(self, kernel_size=15):
        self.kernel_size = kernel_size
    def __call__(self, img):

        kernel = np.zeros((self.kernel_size, self.kernel_size))
        kernel[int((self.kernel_size-1)/2), :] = np.ones(self.kernel_size)
        kernel = kernel / self.kernel_size
        
        return img.filter(ImageFilter.BoxBlur(radius=2))

class SpeckleNoise:
    def __init__(self, std=0.1):
        self.std = std
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + tensor * noise, 0, 1)

def get_domain_transform(domain_id, base_size=256, crop_size=224):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    base_transform = [
        transforms.Resize(base_size),
        transforms.CenterCrop(crop_size),
    ]
    
    domain_configs = {
    0: {
        'name': 'Gaussian Noise (Weak)',
        'description': '弱高斯噪声 (加大 std)',
        'transforms': [],

        'post_tensor': [GaussianNoise(std=0.2)], 
    },
    1: {
        'name': 'Gaussian Noise (Medium)',
        'description': '中等高斯噪声 (加大 std)',
        'transforms': [],

        'post_tensor': [GaussianNoise(std=0.22)], 
    },
    2: {
        'name': 'Gaussian Noise (Strong)',
        'description': '强高斯噪声 (加大 std)',
        'transforms': [],

        'post_tensor': [GaussianNoise(std=0.24)], 
    },
    3: {
        'name': 'Salt-Pepper Noise',
        'description': '椒盐噪声 (加大比例)',
        'transforms': [],

        'post_tensor': [SaltPepperNoise(prob=0.2)], 
    },
    4: {
        'name': 'Salt-Pepper Noise',
        'description': '椒盐噪声 (加大比例)',
        'transforms': [],

        'post_tensor': [SaltPepperNoise(prob=0.22)], 
    },
    5: {
        'name': 'Salt-Pepper Noise',
        'description': '椒盐噪声 (加大比例)',
        'transforms': [],

        'post_tensor': [SaltPepperNoise(prob=0.18)], 
    },
    6: {
        'name': 'Gaussian Noise (Strong)',
        'description': '强高斯噪声 (加大 std)',
        'transforms': [],

        'post_tensor': [GaussianNoise(std=0.5)], 
    },
    7: {
        'name': 'Gaussian Noise (Weak)',
        'description': '弱高斯噪声 (加大 std)',
        'transforms': [],

        'post_tensor': [GaussianNoise(std=0.25)], 
    },
    8: {
        'name': 'Gaussian Noise (Medium)',
        'description': '中等高斯噪声 (加大 std)',
        'transforms': [],

        'post_tensor': [GaussianNoise(std=0.35)], 
    },
    9: {
        'name': 'Gaussian Noise (Strong)',
        'description': '强高斯噪声 (加大 std)',
        'transforms': [],

        'post_tensor': [GaussianNoise(std=0.4)], 
    },
}
    
    config = domain_configs[domain_id]
    
    transform_list = base_transform.copy()
    transform_list.extend(config['transforms'])
    transform_list.append(transforms.ToTensor())
    
    if 'post_tensor' in config:
        transform_list.extend(config['post_tensor'])
    
    transform_list.append(normalize)
    
    final_transform = transforms.Compose(transform_list)
    
    return final_transform, config['name'], config['description']

def visualize_domain_transforms(image_path, save_path='domain_transforms_visualization.png'):
    import matplotlib.pyplot as plt

    original_img = Image.open(image_path).convert('RGB')
    original_img = original_img.resize((224, 224))
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Domain Transformations: All Noise & Compression', 
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for domain_id in range(10):
        transform, name, description = get_domain_transform(domain_id)
        
        if domain_id == 0:

            img_tensor = transforms.ToTensor()(original_img)
        else:

            img = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224)
            ])(original_img)
            
            transform_no_normalize = transforms.Compose([
                t for t in transform.transforms 
                if not isinstance(t, transforms.Normalize)
            ])
            img_tensor = transform_no_normalize(img)
        
        if isinstance(img_tensor, torch.Tensor):
            img_display = img_tensor.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)
        else:
            img_display = np.array(img_tensor) / 255.0
        
        axes[domain_id].imshow(img_display)
        axes[domain_id].set_title(f'D{domain_id}: {name}', fontsize=10, fontweight='bold')
        axes[domain_id].axis('off')
        
        axes[domain_id].text(0.5, -0.1, description, 
                            transform=axes[domain_id].transAxes,
                            ha='center', va='top', fontsize=8, 
                            wrap=True, style='italic')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Domain transforms visualization saved to {save_path}")
    plt.close()

def print_domain_summary():
    print("="*80)
    print("DOMAIN CONFIGURATIONS - ALL NOISE & COMPRESSION")
    print("="*80)
    for domain_id in range(10):
        _, name, description = get_domain_transform(domain_id)
        print(f"\nDomain {domain_id}: {name}")
        print(f"  Description: {description}")
        print(f"  Expected Performance: ~30-60% (before adaptation)")
    
    print("\n" + "="*80)
    print("All domains use degradation:")
    print("  - Sensor noise: Gaussian (3 levels), Poisson, Salt-Pepper, Speckle")
    print("  - Compression: JPEG at quality 15 and 30")
    print("  - Blur: Gaussian blur, Motion blur")
    print("  - NO clean baseline - all domains are corrupted")
    print("="*80)

if __name__ == "__main__":

    print_domain_summary()
    
    print("\nTo visualize domain transforms, run:")
    print("  python domain_transforms.py --image path/to/test/image.jpg")
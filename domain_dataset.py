import torch
from torch.utils.data import Dataset
from domain_transforms import get_domain_transform
class DomainDataset(Dataset):
    def __init__(self, base_dataset, domain_id):
        self.base_dataset = base_dataset
        self.domain_id = domain_id

        self.domain_transform, self.domain_name, self.domain_desc = get_domain_transform(domain_id)
        
        print(f"✓ Domain {domain_id} dataset created: {self.domain_name}")
        print(f"  Description: {self.domain_desc}")
        print(f"  Samples: {len(self.base_dataset)}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):

        from PIL import Image
        from torch.utils.data import Subset
        
        if isinstance(self.base_dataset, Subset):

            actual_idx = self.base_dataset.indices[idx]
            img_path, label = self.base_dataset.dataset.samples[actual_idx]
        else:

            img_path, label = self.base_dataset.samples[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        img_tensor = self.domain_transform(img)
        
        return img_tensor, label

def create_domain_dataloaders(data_path, partition_info, domain_id, batch_size=32, workers=4):
    from torchvision import datasets
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    from create_noniid_partition import get_group_subset_indices
    all_indices = get_group_subset_indices(partition_info, domain_id, data_path)
    
    base_dataset = datasets.ImageFolder(data_path)
    
    domain_subset = Subset(base_dataset, all_indices)
    
    domain_dataset = DomainDataset(domain_subset, domain_id)
    
    np.random.seed(42)
    train_size = int(0.8 * len(all_indices))
    train_indices = list(range(train_size))
    
    train_subset = Subset(domain_dataset, train_indices)
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=workers, 
        pin_memory=True
    )
    
    full_val_loader = DataLoader(
        domain_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    
    return train_loader, full_val_loader

if __name__ == "__main__":
    print("Domain Dataset Wrapper")
    print("="*80)
    print("This module provides domain-aware data loading:")
    print("  - Each domain has different image transformations")
    print("  - Simulates real-world domain shift scenarios")
    print("  - Train/val split: 80%/100% (val uses full domain data)")
    print("  - NO label remapping - uses original ImageNet labels")
    print("="*80)
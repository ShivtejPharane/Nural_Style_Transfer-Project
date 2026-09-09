from torch.utils.data import Dataset
import os
from PIL import Image   
from torchvision import transforms
from torch.utils.data import DataLoader
class ImageFolderDataset(Dataset):
    def __init__(self,root:str,transform):
        super(ImageFolderDataset,self).__init__()
        self.root=root
        self.transform=transform
        self.files=list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        image_path = os.path.join(self.root,self.files[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image

def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize((size,size)))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def adaptive_instance_normalization(content_feats,style_feats):
    #[batch_size,channels,h,w]
    size = content_feats.size()
    style_mean,style_std = calc_mean_std(style_feats)
    content_mean,content_std = calc_mean_std(content_feats)
    normalized_content_feat = (content_feats - content_mean.expand(size))/content_std.expand(size)
    return normalized_content_feat*style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(features,eps=1e-5):
    #[batch_size,channels,h,w]
    size = features.size()
    assert (len(size)==4)
    batch_size,channels = size[:2]
    feat_mean = features.view(batch_size,channels,-1).mean(dim=2).view(batch_size,channels,1,1)
    feat_var = features.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size,channels,1,1)
    return feat_mean,feat_std
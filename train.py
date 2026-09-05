import argparse
import torch
from pathlib import Path
from utils.utils import *
from torch.utils.data import DataLoader


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir',type=str,default=r'E:\Project01\Nural_Style_Transfer-Project\content_data',help='Location of content dataset')
    parser.add_argument('--style_dir',type=str,default=r'E:\Project01\Nural_Style_Transfer-Project\style_data',help='Location of style Dataset')
    parser.add_argument('--vgg',type=str,default=r'E:\Project01\Nural_Style_Transfer-Project\style_data',help='Location of the Pretrained VGG')
    parser.add_argument('--expriment',type=str,default='expriment1',help='Nmae of Expriment')
    parser.add_argument('--final_size',type=int,default=256,help='Size of Final image')
    parser.add_argument('--content_size',type=int,default=512,help='Size of the Content Image')
    parser.add_argument('--style_size',type=int,default=512,help='Size of the Style Image')
    parser.add_argument('--crop',action='store_true',help='Crop Image')
    parser.add_argument('--batch_size',type=int,default=4,help='Size of batches in dataloder')
    return parser.parse_args()
def main():
    args = parse_arguments()

    device=''
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'

    save_dir = Path('expriment')/args.expriment
    save_dir.mkdir(exist_ok=True,parents=True)

    #Save the Arguments values
    with open(save_dir / 'args.txt','w') as args_files:
        for key,value in vars(args).items():
            args_files.write(f'{key}:{value}\n')

    content_transform = get_transform(args.content_size,args.crop,args.final_size)
    style_transform =get_transform(args.style_size,args.crop,args.final_size)

    content_dataset = ImageFolderDataset(args.content_dir,content_transform)
    style_dataset = ImageFolderDataset(args.style_dir,style_transform)

    content_dataloader = DataLoader(content_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)
    style_dataloader = DataLoader(style_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=True,drop_last=True)

    print(f'before batches {len(content_dataset)} after batches {len(content_dataloader)}')
    print(f'before batches {len(style_dataset)} after batches {len(style_dataloader)}')

    for batch in content_dataloader:
        print(batch.shape)

    
if __name__=='__main__':
    main()
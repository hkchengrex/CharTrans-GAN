from net import Net
from font_loader import FontDataset

import torch
import torchvision
import torchvision.transforms as transforms

font_root = './data/limited/'
std_font = './data/standard/mingliu.TTF'
font_size = 52
image_size = 48

char_list_path = './character_set/character_set_1798'

# Load the set of common characters
with open(char_list_path, 'r') as char_set:
    char_list = char_set.readlines()
    char_list = [x.strip() for x in char_list] 

train_batch_size = 128

# Create dataset and dataloader
train_dataset = FontDataset(font_root, char_list, std_font, font_size, image_size)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                          shuffle=True, num_workers=2)

net = Net(train_loader, 500, 10, 'model/BBResize')
net.train()

from net import Net
from font_loader import FontDataset

import torch
import torchvision
import torchvision.transforms as transforms

font_root = './data/test1/'
std_font = './data/standard/'
font_size = 52
image_size = 48

numTransform = 5
numRef = 5

char_list_path = './character_set/character_set_1798'

# Load the set of common characters
with open(char_list_path, 'r') as char_set:
    char_list = char_set.readlines()
    char_list = [x.strip() for x in char_list] 

# char_list = ['的']

train_batch_size = 1

# Create dataset and dataloader
train_dataset = FontDataset(font_root, char_list, std_font, font_size, image_size, numTransform, numRef, '鬼')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                          shuffle=True, num_workers=1)

net = Net(train_loader, 500, 10, 'model/BBResize_S5T5_eN3_NoRefA_train_T5', numTransform, numRef)

net.load(389)
net.generateOne()

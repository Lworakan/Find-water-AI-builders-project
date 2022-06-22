import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
import streamlit as st

from PIL import Image

def load_image(image_file):
	img = Image.open(image_file)
	return img

st.title('Water LineFinder Detection!')

image_file = st.file_uploader("อัพโหลดรูปภาพ", type=["png","jpg","jpeg"])

torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

size = (256, 256)

class LoadData(Dataset):
    def __init__(self, images_path, masks_path):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.len = len(images_path)
        self.transform = transforms.Resize(size)

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx])
        img = self.transform(img)
        img = np.transpose(img, (2, 0, 1))
        img = img/255.0
        img = torch.tensor(img)

        mask = Image.open(self.masks_path[idx]).convert('L')
        mask = self.transform(mask)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = torch.tensor(mask)

        return img, mask
    
    def __len__(self):
        return self.len


if image_file is not None:
    file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
    st.image(load_image(image_file),width=250)
    with open(os.path.join('uploaded_folder',image_file.name),'wb') as f:
        f.write((image_file).getbuffer())
        st.success("File Saved!")

    X = sorted(glob.glob(f'uploaded_folder/{image_file.name}'))
    Y = sorted(glob.glob(f'Masks/{image_file.name}'))

    train_X = X
    train_Y = Y
    valid_X = X
    valid_Y = Y

    train_dataset = LoadData(train_X, train_Y)
    valid_dataset = LoadData(valid_X, valid_Y)

    img, mask = train_dataset[0]

    img.shape
    torch.Size([3, 256, 256])

    class conv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)

            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.relu = nn.ReLU()
    
        def forward(self, images):
            x = self.conv1(images)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            
            return x

    class encoder(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.conv = conv(in_channels, out_channels)
            self.pool = nn.MaxPool2d((2,2))

        def forward(self, images):
            x = self.conv(images)
            p = self.pool(x)

            return x, p

    class decoder(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
            self.conv = conv(out_channels * 2, out_channels)

        def forward(self, images, prev):
            x = self.upconv(images)
            x = torch.cat([x, prev], axis=1)
            x = self.conv(x)

            return x

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()

            self.e1 = encoder(3, 64)
            self.e2 = encoder(64, 128)
            self.e3 = encoder(128, 256)
            self.e4 = encoder(256, 512)

            self.b = conv(512, 1024)

            self.d1 = decoder(1024, 512)
            self.d2 = decoder(512, 256)
            self.d3 = decoder(256, 128)
            self.d4 = decoder(128, 64)

            self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        def forward(self, images):
            x1, p1 = self.e1(images)
            x2, p2 = self.e2(p1)
            x3, p3 = self.e3(p2)
            x4, p4 = self.e4(p3)

            b = self.b(p4)
        
            d1 = self.d1(b, x4)
            d2 = self.d2(d1, x3)
            d3 = self.d3(d2, x2)
            d4 = self.d4(d3, x1)

            output_mask = self.output(d4)
            output_mask = torch.sigmoid(output_mask)

            return output_mask

    batch_size = 8
    num_epochs = 20
    lr = 1e-4
    checkpoint_path = "checkpoint.pth"

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    device = torch.device('cuda')
    model = UNet()
    model = model.to(device)

    the_model = torch.load("water.pt")

    m = UNet()
    m.load_state_dict(torch.load(checkpoint_path))
    m = m.to(device)

if st.button('Click here to Start Prediction'):
    transform = transforms.ToPILImage()
    pred = []
    for x, y in valid_loader:
        image0 = transform(x[0])

        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
    
        y_pred = m(x)
        img = y_pred.cpu().detach().numpy()
        plt.figure(figsize=(30,8))
        plt.imshow(np.squeeze(img), cmap='gray')
        plt.savefig("predicted.jpg", bbox_inches='tight', pad_inches=0)

    st.image(load_image("predicted.jpg"),width=250)
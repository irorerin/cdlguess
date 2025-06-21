import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import argparse
import OpenImageIO as oiio

def read_image_file(file_path, header_only = False):
    result = {'spec': None, 'image_data': None}
    inp = oiio.ImageInput.open(file_path)
    if inp :
        spec = inp.spec()
        result['spec'] = spec
        if not header_only:
            channels = spec.nchannels
            result['image_data'] = inp.read_image(0, 0, 0, channels)
        inp.close()
    return result

def clear_lines(n=2):
    """Clears a specified number of lines in the terminal."""
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)

import signal
def create_graceful_exit():
    def graceful_exit(signum, frame):
        exit(0)
    return graceful_exit
signal.signal(signal.SIGINT, create_graceful_exit())

class CDL(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Each parameter has shape (1, 3, 1, 1) for per-channel transformation
        self.slope = nn.Parameter(torch.ones(1, 3, 1, 1))   # Initialized to 1
        self.offset = nn.Parameter(torch.zeros(1, 3, 1, 1)) # Initialized to 0
        self.power = nn.Parameter(torch.ones(1, 3, 1, 1))   # Initialized to 1

    def forward(self, x):
        x = x * self.slope.clamp(min=1e-8) + self.offset
        x = torch.pow(x, self.power.clamp(min=1, max=1))
        return x
        # return torch.pow(x * self.slope + self.offset, self.power)

def main():
    parser = argparse.ArgumentParser(description="A command-line app that takes source, target, and optional learning rate.")
    parser.add_argument("source", type=str, help="Path to the source file")
    parser.add_argument("target", type=str, help="Path to the target file")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")

    args = parser.parse_args()

    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print(f"Learning Rate: {args.lr}")
    
    img0 = read_image_file(args.source)['image_data'][:, :, :3]
    img1 = read_image_file(args.target)['image_data'][:, :, :3]

    '''
    import matplotlib.pyplot as plt
    plt.imshow(img1)
    plt.show()
    '''

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    img0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).to(device)
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)

    img0 = F.interpolate(
        img0,
        size=(256, 256),
        mode='bilinear',
        align_corners=False, 
        antialias=True
    )

    img1 = F.interpolate(
        img1,
        size=(256, 256),
        mode='bilinear',
        align_corners=False, 
        antialias=True
    )

    cdl = CDL().to(device)
    loss_l1 = nn.L1Loss()

    optimizer = optim.AdamW(cdl.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    
    batch = 0
    cdl.train()

    print ('\n'*3)

    while True:
        optimizer.zero_grad()
        res = cdl(img0)
        loss = loss_l1(res, img1)
        loss.backward()
        optimizer.step()

        clear_lines(1)
        print(loss.item())

if __name__ == "__main__":
    main()


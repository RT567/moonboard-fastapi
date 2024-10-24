import os
import sys
import argparse
from PIL import Image
import torch
import torchvision.models as models
from torchvision.models.googlenet import GoogLeNet_Weights
from torchvision import transforms, io

# takes a list of holds, creates an image from that list, and then classifies the created image

# example usage:
# from 
# python E:\not-messy\software-development\moonboard-classifier\python-image-creator\grade-singular-climb-from-holdlist.py A1 A2

# updated 08/07/2024
# finished writing it, possible some issues with the image handling and transformations
# seems to grade too much stuff as V3, need to test better...?
# potentially images are not created the correct way for the model expects

script_dir = os.path.dirname(os.path.abspath(__file__))
background_image_path = os.path.join(script_dir, "mb2019.jpg")
blue_ring_path = os.path.join(script_dir, "blue-ring-thick.png")
red_ring_path = os.path.join(script_dir, "red-ring-thick.png")
green_ring_path = os.path.join(script_dir, "green-ring-thick.png")
model_params_path = os.path.join(script_dir, "params", "epoch_13.pth")

if len(sys.argv) < 3:
    print("\nUsage:\n > python grade-singular-climb-from-holdlist.py <list of holds>")
    print("\nExample:\n > python E:\\not-messy\\software-development\\moonboard-classifier\\python-image-creator\\grade-singular-climb-from-holdlist.py A1 A2\n")
    sys.exit(1)

# takes a hold and returns the appropriate coord to place it
def hold_to_coords(hold):
    col = hold[0]
    row = int(hold[1:]) 
    x = 59 + ((ord(col)) - ord('A')) * 51   # convert letter to number and scale
    y = 51 + (18 - row) * 51.3                # scale y coordinate
    return int(x), int(y)

# define the pytorch "device" paramter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.googlenet(weights = None, aux_logits=False, init_weights=True)
num_classes = 10 # V3 to V12+

model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)

# file_path = r'.\python-image-creator\params\epoch_13.pth'

if os.path.exists(model_params_path):
    model.load_state_dict(torch.load(model_params_path))
    model.eval()
else:
    print("Cannot find parameter file location as specified in the code...")
    sys.exit(1)

transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.406], std=[0.229, 0.224, 0.225])
])

#create the image file
def hold_to_coords(hold):
    col = hold[0]
    row = int(hold[1:])
    x = 59 + ((ord(col)) - ord('A')) * 51   # convert letter to number and scale
    y = 51 + (18 - row) * 51.3                # scale y coordinate
    return int(x), int(y)

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("holds", type=str, nargs='+')
    return parser.parse_args()

args = parse_args()
holds = args.holds

print(holds)

ring_size = (76, 76)  # Set the desired size for the rings

background = Image.open(background_image_path)
blue = Image.open(blue_ring_path).resize(ring_size)
red = Image.open(red_ring_path).resize(ring_size)
green = Image.open(green_ring_path).resize(ring_size)

background_copy = background.copy()

green_quota = 2

for hold in holds:
    x, y = hold_to_coords(hold)
    
    if y == 51:
        background_copy.paste(red, (x, y), red)
    elif y > 660 and green_quota > 0:
        background_copy.paste(green, (x, y), green)
        green_quota -= 1
    else:
        background_copy.paste(blue, (x,y), blue)

width, height = background_copy.size

background_copy = background_copy.crop((0, 10, width - 0, height - 20)) # L Top left Bottom

resized = background_copy.resize((224, 224))
resized.show()

image = transform(resized).unsqueeze(0).to(device)  # Transform and add batch dimension

with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    grade = f"V{predicted.item() + 3}"
    result = f"Grade predicted as: {grade}"
    print(result)

print("finishing script")
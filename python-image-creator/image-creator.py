import argparse
import os
from PIL import Image

# script is used to create the images depending on input arguments
# input arguments are describing hold positions
# and place them in a certain directory 
# example usage: (from directory containing the script)
# python image-creator.py A1 A18 B18 C18 D18 E18 F18 G18 H18 I18 J18 K18 A17 B17 C17 D17 A9 B9 C9 K1 A6 I5 J6 I8 J10

# takes a hold and returns the appropriate coord to place it
def hold_to_coords(hold):
    col = hold[0]
    row = int(hold[1:]) 
    x = 59 + ((ord(col)) - ord('A')) * 51   # convert letter to number and scale
    y = 51 + (18 - row) * 51.3                # scale y coordinate
    return int(x), int(y)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("holds", type=str, nargs='+')
    return parser.parse_args()

args = parse_args()
name_given = args.name
holds = args.holds

print(holds)

print(hold_to_coords(holds[0]))

background = Image.open("mb2019.jpg")

# Open and resize the ring images
ring_size = (76, 76)  # Set the desired size for the rings
blue = Image.open("blue-ring-thick.png").resize(ring_size)
red = Image.open("red-ring-thick.png").resize(ring_size)
green = Image.open("green-ring-thick.png").resize(ring_size)

background_copy = background.copy()

green_quota = 2

for hold in holds:
    x, y = hold_to_coords(hold)
    
    if y == 51:
        background_copy.paste(red, (x, y), red)
    elif y > 660 and green_quota > 0:
        background_copy.paste(green, (x, y), green)
        green_quota = green_quota - 1
    else:
        background_copy.paste(blue, (x, y), blue)

width, height = background_copy.size
background_copy = background_copy.crop((0, 10, width - 0, height - 20)) # L Top left Bottom

resized = background_copy.resize((224,224))
# resized.show()
#background_copy.show()

output_dir = r'E:\not-messy\software-development\moonboard-classifier\python-image-creator\created-images'
result_image_path = os.path.join(output_dir, name_given)

resized.save(result_image_path)

print("Script reached end")
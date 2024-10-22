import argparse
import os
import csv
from PIL import Image

# Script to create images based on CSV input

row_offset = 51

# Takes a hold and returns the appropriate coord to place it
def hold_to_coords(hold):
    col = hold[0]
    row = int(hold[1:])
    x = 59 + ((ord(col)) - ord('A')) * 51   # Convert letter to number and scale
    y = row_offset + (18 - row) * 51.3      # Scale y coordinate
    return int(x), int(y)

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file")
    parser.add_argument("output_dir", type=str, help="Path to the output directory")
    return parser.parse_args()

# Check if the name is valid
def is_valid_name(name):
    return name and not name.startswith('#')

# Main function to process the CSV and create images
def main():
    args = parse_args()
    csv_file = args.csv_file
    output_dir = args.output_dir

    # Open the background image and the ring images
    background = Image.open("mb2019.jpg")
    ring_size = (76, 76)  # Set the desired size for the rings
    blue = Image.open("blue-ring-thick.png").resize(ring_size)
    red = Image.open("red-ring-thick.png").resize(ring_size)
    green = Image.open("green-ring-thick.png").resize(ring_size)

    name_counter = 1  # Initialize a counter for naming

    with open(csv_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            print(f"Read row: {row}")  # Debugging: print each row

            if len(row) < 3:  # Ensure the row has at least 3 columns (name, grade, and one hold)
                print(f"Skipping incomplete row: {row}")
                continue

            name_given = row[0]
            grade = row[1]
            holds = [hold for hold in row[2:] if hold]

            if not is_valid_name(name_given):
                name_given = str(name_counter)
                name_counter += 1

            background_copy = background.copy()
            green_quota = 2

            for hold in holds:
                x, y = hold_to_coords(hold)

                if y == row_offset:
                    background_copy.paste(red, (x, y), red)
                elif y > 660 and green_quota > 0:
                    background_copy.paste(green, (x, y), green)
                    green_quota = green_quota - 1
                else:
                    background_copy.paste(blue, (x, y), blue)

            width, height = background_copy.size
            background_copy = background_copy.crop((0, 10, width - 0, height - 20))  # L Top left Bottom
            resized = background_copy.resize((224, 224))

            output_name = f"{name_given}_GRADE={grade}.png"
            result_image_path = os.path.join(output_dir, output_name)
            resized.save(result_image_path)

    print("Script reached end")

if __name__ == "__main__":
    main()

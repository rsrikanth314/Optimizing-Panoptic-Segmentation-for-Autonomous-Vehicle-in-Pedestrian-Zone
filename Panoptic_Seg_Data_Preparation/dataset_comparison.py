import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def count_pixels_per_color(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    # Reshape the array to a 2D array of shape (height * width, 3)
    reshaped_array = image_array.reshape(-1, 3)
    
    # Get the unique color combinations and their counts
    unique_colors, counts = np.unique(reshaped_array, axis=0, return_counts=True)
    
    # Create a dictionary to store the color counts
    color_counts = {tuple(color): count for color, count in zip(unique_colors, counts)}
    
    return color_counts

# Example usage for counting pixels of each color in multiple images in a directory
directory = "/home/yalamaku/Documents/Thesis/Dataset_files/ZED_camera_dataset/Roboflow_annotated_data/CustomDataset/train/train_coco_panoptic"

color_counts = {}
counter = 0
for filename in os.listdir(directory):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        counter+=1
        image_path = os.path.join(directory, filename)
        image_color_counts = count_pixels_per_color(image_path)
        
        # Add the pixel counts for each color to the overall color_counts dictionary
        for color, count in image_color_counts.items():
            if color in color_counts:
                color_counts[color] += count
            else:
                color_counts[color] = count

# What are the number of pixels in the dataset:
num_pixels_imgs = counter*(800*400)
print("Pixel counts for each color:")
print("Number of images: ", counter)
for color, count in color_counts.items():
    color_counts[color] = (count/num_pixels_imgs)*100
    print(f"Color {color}: {(count/num_pixels_imgs)*100} pixels")

print(color_counts)

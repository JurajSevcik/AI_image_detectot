import os
from datasets import load_dataset

# Define the folder where you want to save the images
output_directory = "real_images"
os.makedirs(output_directory, exist_ok=True)

# Load the dataset from Hugging Face
print("Downloading and loading the dataset...")
dataset = load_dataset("bitmind/bm-real", split="train")

print(f"Dataset loaded! Total images to save: {len(dataset)}")

# Iterate through the dataset and save each image
for index, item in enumerate(dataset):
    image = item['image']  # The PIL image object
    #img_id = item['id']  # The unique ID from the dataset

    # Define the save path
    save_path = os.path.join(output_directory, f"real_{index}.png")

    # Save the image
    image.save(save_path)

    if index % 1000 == 0:
        print(f"Saved {index} images...")

print(f"All images have been successfully saved to the '{output_directory}' folder!")
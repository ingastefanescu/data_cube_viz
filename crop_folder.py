from PIL import Image
import os

def crop_and_save_image(image_path, crop_coords, output_folder):
    image = Image.open(image_path)
    cropped_image = image.crop(crop_coords)
    
    # Ensure the cropped image is in portrait orientation
    width, height = cropped_image.size
    if width > height:
        cropped_image = cropped_image.transpose(Image.ROTATE_90)
    
    # Extract original filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save cropped image to new file
    cropped_filename = os.path.join(output_folder, f"{filename}_cropped.jpg")
    cropped_image.save(cropped_filename)
    
    print(f"Cropped image saved as: {cropped_filename}")

def main():
    # Define the path to your images folder
    folder_path = r"D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\msi"
    
    # Define crop coordinates
    x1, x2 = 500, 600  # horizontal boundaries
    y1, y2 = 1950, 2100  # vertical boundaries
    crop_coords = (x1, y1, x2, y2)
    
    # Create output folder in the parent directory
    parent_folder = os.path.dirname(folder_path)
    output_folder_name = os.path.basename(folder_path) + "_cropped"
    output_folder = os.path.join(parent_folder, output_folder_name)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through each image file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust file extensions as necessary
            image_path = os.path.join(folder_path, filename)
            crop_and_save_image(image_path, crop_coords, output_folder)

if __name__ == "__main__":
    main()

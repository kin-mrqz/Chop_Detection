import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_convert_image(image_path):
    # Read the image in color mode
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or unable to read: {image_path}")
    # Convert the image from BGR to YCrCb color space
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

def binarize_components(image_ycbcr):
    # Split the image into Y, Cb, and Cr components
    Y, Cb, Cr = cv2.split(image_ycbcr)
    # Apply adaptive thresholding to binarize the Cb and Cr components
    binary_Cb = cv2.adaptiveThreshold(Cb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    binary_Cr = cv2.adaptiveThreshold(Cr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return binary_Cb, binary_Cr

def combine_binarized_components(binary_Cb, binary_Cr):
    # Analyze the average values to decide on combining method
    avg_Cb = np.mean(binary_Cb)
    avg_Cr = np.mean(binary_Cr)
    if (avg_Cb < 15 and avg_Cb > 3) or (avg_Cr > 240 and avg_Cr < 252):
        # Apply adjustment based on component intensities
        if avg_Cb < 15:
            return np.minimum(binary_Cb, binary_Cr)
        else:
            return np.maximum(binary_Cb, binary_Cr)
    return cv2.bitwise_or(binary_Cb, binary_Cr)

def display_images(original, binary_Cb, binary_Cr, combined_binary):
    # Convert the original image to RGB for displaying
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.subplot(221), plt.imshow(original_rgb), plt.title('Original Image'), plt.axis('off')
    plt.subplot(222), plt.imshow(binary_Cb, cmap='gray'), plt.title('Binarized Blue Component (Cb)'), plt.axis('off')
    plt.subplot(223), plt.imshow(binary_Cr, cmap='gray'), plt.title('Binarized Red Component (Cr)'), plt.axis('off')
    plt.subplot(224), plt.imshow(combined_binary, cmap='gray'), plt.title('Combined Binarized Components'), plt.axis('off')
    plt.tight_layout()
    plt.show()

def process_and_save_images(source_directory, target_directory):
    # Ensure target directory exists
    os.makedirs(target_directory, exist_ok=True)
    # Process each image file in the source directory
    for filename in os.listdir(source_directory):
        file_path = os.path.join(source_directory, filename)
        try:
            image_ycbcr = load_and_convert_image(file_path)
            binary_Cb, binary_Cr = binarize_components(image_ycbcr)
            combined_binary = combine_binarized_components(binary_Cb, binary_Cr)
            target_path = os.path.join(target_directory, filename)
            cv2.imwrite(target_path, combined_binary)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    source_dir = "sample"
    target_dir = "processed"
    process_and_save_images(source_dir, target_dir)

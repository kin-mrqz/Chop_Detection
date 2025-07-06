import os
import cv2
import xml.etree.ElementTree as ET
import csv

def parse_xml(xml_file):
    """
    Parse the given XML file to extract bounding boxes and labels.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    for member in root.findall('object'):
        labels = member.find('name').text
        xmin = int(member.find('bndbox/xmin').text)
        ymin = int(member.find('bndbox/ymin').text)
        xmax = int(member.find('bndbox/xmax').text)
        ymax = int(member.find('bndbox/ymax').text)
        annotations.append((labels, (xmin, ymin, xmax, ymax)))
    return annotations

def process_directory(image_dir, xml_dir, output_csv_path):
    """
    Process the directory of images, pair them with their corresponding XML annotations,
    and write the results to a CSV file.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Label", "XMin", "YMin", "XMax", "YMax", "ImagePath"])

        for filename in os.listdir(image_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_path = os.path.join(image_dir, filename)
                xml_path = os.path.join(xml_dir, os.path.splitext(filename)[0] + '.xml')

                if not os.path.exists(xml_path):
                    print(f"XML file does not exist for {filename}, skipping...")
                    continue

                annotations = parse_xml(xml_path)
                for label, bbox in annotations:
                    writer.writerow([filename, label, bbox[0], bbox[1], bbox[2], bbox[3], image_path])

                print(f"Processed {filename} with annotations: {annotations}")

# Directory paths setup
image_directory = "processed" # replace with path to preprocessed images in YCbCr binarized formm
xml_directory = "labelled" # path to labelled annotations (used labelImg) provided
output_csv_file = "annotations.csv"


# Run the processing function
process_directory(image_directory, xml_directory, output_csv_file)

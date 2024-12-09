# Defect Image Creation and Processing

## Project Overview
The main goal of this project is to create and process images with different types of defects. This involves overlaying plastic images onto dataset images to simulate defects, adding transformations, and saving them along with corresponding information such as bounding boxes and color details. The project also includes logging and organizing this data into CSV, Excel, and JSON files for further analysis.

## Key Parts of the Code

### Setup and Imports
- The project imports various necessary libraries, such as:
  - **PIL** for image handling
  - **os** for file operations
  - **pandas** for handling data logging in CSV and Excel formats
  - **cv2** for image processing

### Image Loading and Processing

#### Directories Setup
- Several directories are defined for:
  - Dataset images
  - Plastic images
  - Output directories (e.g., `output_folder_1`, `output_folder_2`, etc.)

#### Image Selection and Augmentation
- **Random Transformations**:
  - The `random_transformations()` function applies random transformations (e.g., rotation, flipping) to plastic images to make them appear different before adding them as defects.
  
- **Image Overlaying**:
  - The `process_image()` function pastes a plastic image onto a dataset image at a random location. It also applies a gradient transparency to make the defect blend more realistically with the original image.

#### Gradient Mask Creation
- The `create_gradient_mask()` function creates a gradient mask, allowing a smooth transition when overlaying defects onto dataset images.

### Extracting Features from Images

#### Color Extraction
- **Extract Colors**:
  - The `extract_colors()` function uses **ColorThief** to extract the top five colors from an image. This helps in understanding the main features of the defect region.

#### Class and Defect Information
- The `get_image_class_and_defect()` function extracts defect type and class from the image filename, generating metadata for each defected image.

### Translation of Defect Terms
- The `translate_term()` function uses a predefined dictionary to translate defect-related terms into Korean. This can be useful for localized data analysis.

### Saving Image Information

#### Image Saving and Logging
- **Processed Images** are saved in specified output directories.
- **Logs** are created in multiple formats:
  - **CSV and Excel Logs**: These logs include coordinates, color information, and defect classifications.
  - **Bounding Box Logs**: Store bounding box coordinates for each defect overlay, indicating where the defect is located in the image.
  - **JSON Logs**: The `save_json()` function saves metadata about each defected image in JSON format, which may be used for machine learning training or other purposes.

### Renaming and Cleaning Files

#### Renaming Images
- The `rename_images()` function renames images in specified directories by keeping only relevant parts of their names. This helps organize the data more cleanly.

#### Updating JSON File Names
- The `update_image_name_in_json_file()` function updates image names inside JSON files to match the cleaned filenames, ensuring consistency across different logs and data formats.

---

## Usage

1. **Install Dependencies**: Ensure you have the required libraries installed, such as PIL, OpenCV, pandas, and ColorThief.
2. **Data Preparation**: Place the dataset and plastic images in the appropriate directories.
3. **Run Processing**:
   - Use the functions provided to simulate defects, process images, and save metadata.
   - Logs will be generated in CSV, Excel, and JSON formats for further analysis.

---

## Results
- The project creates defected images with detailed metadata, including bounding box coordinates, color details, and defect classifications.
- The processed images and logs can be used for further analysis or as part of a machine learning training dataset.

---

## Acknowledgments
- The deep learning and computer vision community for the tools and libraries that facilitated this project.

1

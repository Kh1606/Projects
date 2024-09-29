![alt text](https://github.com/user-attachments/assets/32e51fae-0ef3-4188-8410-27ef9751ed4d)

# Insomnia Data Analysis Project

This project aims to analyze sleep data related to insomnia. The data is processed from images, using Python scripts to extract relevant information and generate insights.

## Project Description
This project involves analyzing sleep duration and stages, primarily using image files to extract data through OCR. The extracted data is further processed to generate CSV files for comprehensive analysis. The goal is to help understand sleep patterns and potential insomnia indicators by leveraging visual data sources.

## Features
- Extract sleep data from images using OCR tools like EasyOCR.
- Convert image information into structured CSV files.
- Analyze sleep stages to identify insomnia patterns.

### Code Overview
# Sleep Data Analysis and Extraction Tool

This project aims to analyze sleep data by extracting relevant information from images, preprocessing it, and generating CSV files for further analysis. Below, you'll find an overview of the key functionalities.

## Key Parts of the Code

### Importing Libraries
Several Python libraries are imported for various purposes:
- **os**: Used for file management.
- **pytesseract**: For extracting text from images.
- **EasyOCR**: Alternative OCR tool for text extraction.
- **cv2 (OpenCV)**: For image processing.
- **numpy**: For numerical operations.
- **pandas**: For data manipulation and analysis.
- **matplotlib**: For visualizing data.

### File Handling and Image Filtering
- Handles file directories and filters images based on their extensions (e.g., `.jpg`, `.png`).
- Uses **os** for file operations, such as renaming, deleting, and organizing images.

### Extracting Text from Images
- **pytesseract** is used to extract sleep data from images.
- A function converts time from Korean format (오전 or 오후) to a 24-hour format.
- Regular expressions are used to extract time ranges from filenames, which helps in naming output files correctly.

### Image Preprocessing
- Images are preprocessed to remove unnecessary data and isolate the graph area.
- Converted to grayscale and processed using thresholding and contour detection.
- Cropping ensures that only the relevant part of the graph is retained for further analysis.

### Sleep Data Extraction and Storage
- Processes each image to extract sleep data based on different stages (e.g., Awake, Light sleep, Deep sleep).
- The extracted data is stored in a structured format as a CSV file.
- CSV files contain columns such as **Time**, **Class**, and **Awake Status**.

### Time Calculations
- Functions are defined to handle time conversions and calculate the duration between different time intervals.
- The extracted time is used to calculate the image width in minutes, determining the dimensions for resizing.

### Analyzing Sleep Data
- **pandas** is used to create dataframes for storing extracted sleep metrics.
- Analysis includes calculating the total time spent in each sleep stage and identifying periods of wakefulness.
- The output is visualized using **matplotlib** to generate plots representing sleep patterns.

### Image Visualization and Plotting
- A function plots sleep classes over time using **matplotlib**.
- Specific conditions are applied, such as identifying awake moments with 10 or more consecutive non-empty rows.
- Sleep stages (**Deep sleep**, **Light sleep**, **REM sleep**, **Awake**) are plotted, using CSV data for visualization.

### File Management
- File renaming and removal functions help organize input/output folders.
- Ensures that only processed images and corresponding CSV files are retained, removing unmatched files.

## Main Functionalities
- **Text Extraction**: Uses OCR (**pytesseract**) to extract sleep data from images.
- **Image Preprocessing**: Converts, filters, and resizes images to focus on relevant parts.
- **Data Storage**: Saves sleep data in CSV format for further analysis.
- **Data Analysis**: Calculates sleep metrics and visualizes data to identify insomnia patterns.
- **File Operations**: Automates file renaming and cleanup processes to streamline workflow.

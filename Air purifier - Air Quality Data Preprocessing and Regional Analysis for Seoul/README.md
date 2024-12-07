### Air Quality Data Preprocessing and Regional Analysis for Seoul
![alt text](https://github.com/Kh1606/Projects/blob/main/Air%20purifier%20-%20Air%20Quality%20Data%20Preprocessing%20and%20Regional%20Analysis%20for%20Seoul/aqi.gif)


## Project Overview
This project aims to handle and preprocess CSV files containing air quality data from Seoul, specifically focusing on:

- **Renaming and cleaning up filenames**.
- **Splitting regional AQI data** into separate files for each region.
- **Standardizing and transforming** the columns and values within the CSV files.

## Key Parts of the Code

### Import Libraries and Initial Setup
- The project utilizes several libraries to manage data and file operations:
  - **pandas**: For data transformation and analysis.
  - **os**: For handling file paths and operations.
  - **json**: For handling metadata and data configurations.
  
- Paths for input and output folders are defined to store CSV files containing air quality data.

### Data Preprocessing Steps

#### Loading and Renaming Files
- The code loads multiple CSV files containing regional air quality data.
- **Filename Simplification**:
  - Filenames such as `서울시_기간별_시간평균_대기환경_정보_2019.08_real_real.csv` are renamed to `2019.08.csv` by removing redundant parts for ease of access.

#### Column Transformation and Cleanup
- **Operations on CSV Files**:
  - **Deleting Unnecessary Columns**: Columns like `권역코드`, `권역명`, and `측정소코드` are dropped to focus on essential information.
  - **Renaming Columns**: Column names are translated from Korean to English (e.g., `측정일시` is renamed to `date`).
  - **Fixing Encoding Issues**: Certain Korean characters or symbols are replaced to ensure clean data for analysis.

### Splitting Data by Region
- The code splits the dataset by the **region name (`측정소명`)** into separate CSV files for each region in Seoul.
- Each new CSV file is named accordingly (e.g., `종로구_2019.09.csv`), containing records specific to that region and month.
- **Output Files**:
  - Each region-specific file is saved in a designated output folder for easy reference.

### Logging Information
- During preprocessing, messages are printed to indicate progress:
  - The number of records in each new file.
  - Successful saves of newly generated files.
  
## Dataset Summary
- **Number of Rows**: 8,760
- **Number of Columns**: 49

---

## Usage

1. **Install Dependencies**: Ensure you have pandas and other required libraries installed.
2. **Data Preparation**: Place the CSV files in the appropriate input folder.
3. **Run Preprocessing**:
   - Load and rename files.
   - Apply column transformations.
   - Split the data by region and save the outputs.

---

## Results
- Preprocessed and cleaned air quality data, split by regions in Seoul.
- Separate CSV files for each region and month for easy analysis and visualization.

---

## Acknowledgments
- **Seoul Open Data Portal** for providing air quality datasets.
- The deep learning and data science community for their resources on data preprocessing techniques.

---

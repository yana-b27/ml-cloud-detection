# ml-cloud-detection
![icon](https://github.com/user-attachments/assets/e0858142-d82a-4d7c-9a00-5bc350d12b76)
## Overview
The ML Cloud Detection plugin is a QGIS tool designed to identify clouds in satellite imagery using machine learning techniques. It processes .tif files with a minimum of 4 channels (blue, green, red, NIR) and generates a cloud mask as output, which is visualized in QGIS. The plugin leverages two pre-trained models—Logistic Regression and Naive Bayes—allowing users to select the preferred method for cloud detection.

## Features
- Input Support: Processes .tif files with at least 4 channels (blue, green, red, NIR).
- Machine Learning Models: Includes two pre-trained models:
   - Logistic Regression
   - Naive Bayes
- Data Preprocessing:
   - Normalizes input channels to a [0, 1] range.
   - Calculates spectral indices (e.g., NDWI) and custom metrics (e.g., Cloud Index, Haze-Optimized   Transformation) for model input.
- Visualization: Outputs a cloud mask as a raster layer in QGIS
- Additional Options:
   - Option to add the input image as a layer in QGIS.
   - Customizable output file name and directory.

## Limitations
- Requires .tif files with at least 4 channels (blue, green, red, NIR).
- Models are pre-trained and may not generalize well to all types of satellite imagery.

## Dependencies
- QGIS 3.x
- Python libraries:
   - rasterio: For reading and writing .tif files.
   - numpy: For numerical computations.
   - pandas: For feature engineering.
   - scikit-learn: For machine learning models.
   - spyndex: For calculating spectral indices like NDWI.
   - joblib: For loading pre-trained models.

## Directory Structure
Below is the structure of the repository with descriptions of each file and directory:
```
cloud-detection-plugin/
├── i18n/                         # Internationalization files
├── ml_models/                    # Directory containing pre-trained machine learning models
│   ├── logistic_regression.joblib  # Pre-trained Logistic Regression model for cloud detection
│   └── naive_bayes.joblib          # Pre-trained Naive Bayes model for cloud detection
├── __init__.py                   # Initialization file required for QGIS to recognize the plugin
├── cloud_detection.py            # Main plugin file that registers the plugin in QGIS
├── cloud_detection_dialog.py     # Contains the dialog window UI and logic for user interaction
├── cloud_detection_dialog_base.ui  # QGIS UI design file (Qt Designer)
├── cloud_detector.py             # Core logic for image processing and cloud detection
├── icon.png                      # Icon file displayed in the QGIS toolbar for the plugin
├── metadata.txt                  # Metadata file with plugin information (version, author, etc.) for QGIS
├── plugin_upload.py              # Script for uploading to QGIS repository
├── requirements.txt              # List of Python dependencies required to run the plugin
├── resources.qrc                 # Resource file containing references to icons and other assets
├── .gitignore                    # Git ignore file to exclude temporary files (e.g., pycache)
├── LICENSE                       # License file for the project
└── README.md                     # Main documentation file with plugin description
```

## Screenshots
### Plugin Interface
![image](https://github.com/user-attachments/assets/e14cbaf0-e227-4bb0-81bd-b30a3c2bb616)

### Example Result
Before:  
![image](https://github.com/user-attachments/assets/fe4dfb16-5209-4307-ba3a-74293702a7b4)

After:  
![image](https://github.com/user-attachments/assets/6b0dce98-4ee3-4bc3-942b-97285de8ac25)



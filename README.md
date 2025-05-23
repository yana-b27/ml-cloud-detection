# ML Cloud Detection
![icon](https://github.com/user-attachments/assets/e0858142-d82a-4d7c-9a00-5bc350d12b76)
## Overview
The ML Cloud Detection plugin is a QGIS tool designed to identify clouds in satellite imagery using machine learning techniques. It processes .tif files with a minimum of 4 channels (blue, green, red, NIR) and generates a cloud mask as output, which is visualized in QGIS. The plugin leverages two pre-trained models—Logistic Regression and Naive Bayes—allowing users to select the preferred method for cloud detection. The process of training empirical and machine learning models is described in the Jupyter notebook at the following [link](https://github.com/yana-b27/ml-cloud-detection/blob/main/cloud_detection_algorithms.ipynb).

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
├── cloud_detection_algorithms.ipynb  # Jupyter notebook with data preparation and analysis of cloud detection models
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

## Video example

Сlick on the preview below to open the video in Google Drive:

[![Plugin example](./assets/preview.png)](https://drive.google.com/file/d/1qyJnjhdVLgCHIlHvSQmM1MlNA2nyeDvP/view?usp=sharing)

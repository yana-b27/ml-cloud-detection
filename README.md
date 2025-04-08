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
 
## Directory Structure
Below is the structure of the repository with descriptions of each file and directory:
```
cloud-detection-plugin/
├── ml_models/                    # Directory containing pre-trained machine learning models
│   ├── logistic_regression.joblib  # Pre-trained Logistic Regression model for cloud detection
│   └── naive_bayes.joblib          # Pre-trained Naive Bayes model for cloud detection
├── init.py                       # Initialization file required for QGIS to recognize the plugin
├── cloud_detection.py            # Main plugin file that registers the plugin in QGIS
├── cloud_detection_dialog.py     # Contains the dialog window UI and logic for user interaction
├── cloud_detector.py             # Core logic for image processing and cloud detection
├── icon.png                      # Icon file displayed in the QGIS toolbar for the plugin
├── metadata.txt                  # Metadata file with plugin information (version, author, etc.) for QGIS
├── requirements.txt              # List of Python dependencies required to run the plugin
├── resources.qrc                 # Resource file containing references to icons and other assets
├── .gitignore                    # Git ignore file to exclude temporary files (e.g., pycache)
├── LICENSE                       # License file for the project
└── README.md                     # Main documentation file with plugin description
```

"""
Cloud Detection Module
This module provides functionality for detecting clouds in satellite images using
pre-trained machine learning models. It includes classes and methods for image
processing, feature computation, and cloud detection.
Classes:
    ImageDataset:
        A class for processing satellite images and computing various features
        for cloud detection. It supports loading images, normalizing channels,
        computing spectral indices, and preparing preprocessed data for model input.
    CloudDetectionModel:
        A class for handling cloud detection using pre-trained machine learning
        models. It supports logistic regression and naive Bayes models for
        predicting cloud presence in images.
Functions:
    detect_clouds(image_path, output_path, model_type="logistic regression"):
        Detects clouds in a satellite image using a specified model type and saves
        the resulting prediction map to the specified output path.
Dependencies:
    - os: For handling file paths.
    - joblib: For loading pre-trained machine learning models.
    - numpy: For numerical operations on image data.
    - pandas: For handling preprocessed image data as DataFrames.
    - sklearn.preprocessing.MinMaxScaler: For normalizing image data.
    - spyndex: For computing spectral indices like NDWI.
    - rasterio: For reading and writing geospatial raster data.
"""

import os
from joblib import load
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import spyndex
import rasterio


class ImageDataset:
    """
    A class for processing satellite images and computing various features for cloud detection dataset.
    Attributes:
        image_path (str): The file path to the input image.
        image_data (numpy.ndarray): The raw image data loaded from the file.
        image_data_preprocessed (pandas.DataFrame): The preprocessed image data with computed features for cloud detection model.
        profile (dict): Metadata profile of the image.
    Methods:
        __init__(image_path):
            Initializes the ImageDataset class with the given image path.
        load_image():
            Loads the image data from the specified file path and stores it in the `image_data` attribute.
        normalize_channels():
            Normalizes each channel of the image data to the range [0, 1].
        compute_features():
            Computes various features such as NDWI, cloud index, and haze optimized transformation (HOT).
            Prepares a DataFrame with normalized features and additional derived features.
        make_image_2d(image_dataset):
            Reshapes a 2D or 3D image dataset into a 2D array and scales its values to the range [0, 1].
    """

    def __init__(self, image_path):
        """
        Initializes the ImageDataset class with the specified image path.

        Args:
            image_path (str): The file path to the image to be processed.

        Attributes:
            image_path (str): Stores the file path to the image.
            image_data (None): Placeholder for the raw image data, initialized as None.
            image_data_preprocessed (None): Placeholder for the preprocessed image data, initialized as None.
            profile (None): Placeholder for the image profile metadata, initialized as None.
        """
        self.image_path = image_path
        self.image_data = None
        self.image_data_preprocessed = None
        self.profile = None

    def load_image(self):
        """
        Loads an image from the specified file path and reads its data and profile.

        This method uses the `rasterio` library to open the image file located at
        `self.image_path`. It reads the image data into `self.image_data` and
        stores the image's metadata profile in `self.profile`.

        Attributes:
            self.image_data (numpy.ndarray): The image data read from the file.
            self.profile (dict): The metadata profile of the image.
        """
        with rasterio.open(self.image_path) as src:
            self.image_data = src.read()
            self.profile = src.profile

    def normalize_channels(self):
        """
        Normalizes the channels of the image data to a range of [0, 1].
        This method processes each channel of the image data independently. For each channel:
        - The minimum and maximum values are calculated.
        - If the maximum value is greater than the minimum value, the channel is normalized
          by subtracting the minimum value and dividing by the range (max - min).
        - If the maximum and minimum values are equal, the channel remains unchanged.
        The normalized data replaces the original image data.
        Attributes:
            self.image_data (numpy.ndarray): A 3D array representing the image data with
                                             shape (channels, height, width). The data type
                                             is converted to float32 during normalization.
        """

        normalized_data = np.zeros_like(self.image_data, dtype=np.float32)
        for i in range(self.image_data.shape[0]):
            channel = self.image_data[i, :, :]
            channel_min = channel.min()
            channel_max = channel.max()
            if channel_max > channel_min:
                normalized_data[i, :, :] = (channel - channel_min) / (
                    channel_max - channel_min
                )
            else:
                normalized_data[i, :, :] = channel

        self.image_data = normalized_data

    def compute_features(self):
        """
        Computes various spectral features and indices from the input image data, normalizes them,
        and prepares a preprocessed DataFrame for further analysis.
        The method performs the following steps:
        1. Extracts individual spectral bands (blue, green, red, and near-infrared (nir)) from the image data.
        2. Computes the Normalized Difference Water Index (NDWI) using the green and nir bands.
        3. Calculates additional indices:
           - Cloud index: Average of all spectral bands.
           - Haze Optimized Transformation (HOT): Derived from blue and red bands.
        4. Normalizes all computed indices to a range of [0, 1].
        5. Creates new features by combining indices (e.g., blue * ndwi, ndwi * HOT).
        6. Normalizes the combined features again.
        7. Drops specific columns and stores the preprocessed data for further use.
        Attributes:
            self.image_data (numpy.ndarray): The input image data with shape (bands, height, width).
            self.image_data_preprocessed (pandas.DataFrame): The preprocessed DataFrame containing
                normalized features and indices.
        """

        blue = self.image_data[0, :, :]
        blue_1d = self.make_image_2d(blue)

        green = self.image_data[1, :, :]
        green_1d = self.make_image_2d(green)

        red = self.image_data[2, :, :]
        red_1d = self.make_image_2d(red)

        nir = self.image_data[3, :, :]
        nir_1d = self.make_image_2d(nir)

        ndwi = spyndex.computeIndex(
            index=["NDWI"],
            params={"N": self.image_data[3, :, :], "G": self.image_data[1, :, :]},
        )
        ndwi_1d = self.make_image_2d(ndwi)

        cloud_index = (
            self.image_data[0, :, :]
            + self.image_data[1, :, :]
            + self.image_data[2, :, :]
            + self.image_data[3, :, :]
        ) / 4
        cloud_index_1d = self.make_image_2d(cloud_index)

        haze_opt_trans = (
            self.image_data[0, :, :] - 0.5 * self.image_data[2, :, :] - 0.08
        )
        haze_opt_trans_1d = self.make_image_2d(haze_opt_trans)

        indices_array = np.concatenate(
            (
                blue_1d,
                green_1d,
                red_1d,
                nir_1d,
                ndwi_1d,
                cloud_index_1d,
                haze_opt_trans_1d,
            ),
            axis=1,
        )
        indices_df = pd.DataFrame(
            indices_array,
            columns=["blue", "green", "red", "nir", "ndwi", "cloud_index", "HOT"],
        )

        for column in indices_df.columns:
            indices_df[column] = (indices_df[column] - indices_df[column].min()) / (
                indices_df[column].max() - indices_df[column].min()
            )

        indices_df["blue ndwi"] = indices_df["blue"] * indices_df["ndwi"]
        indices_df["green ndwi"] = indices_df["green"] * indices_df["ndwi"]
        indices_df["red ndwi"] = indices_df["red"] * indices_df["ndwi"]
        indices_df["nir ndwi"] = indices_df["nir"] * indices_df["ndwi"]
        indices_df["ndwi HOT"] = indices_df["ndwi"] * indices_df["HOT"]
        indices_df["ndwi cloud_index"] = indices_df["ndwi"] * indices_df["cloud_index"]

        for column in indices_df.columns:
            indices_df[column] = (indices_df[column] - indices_df[column].min()) / (
                indices_df[column].max() - indices_df[column].min()
            )

        self.image_data_preprocessed = indices_df.drop(
            ["green", "red", "nir", "ndwi", "cloud_index"], axis=1
        )

    def make_image_2d(self, image_dataset):
        """
        Transforms a multi-dimensional image dataset into a 2D array and scales its values.
        This function reshapes a 3D or 2D image dataset into a 2D array where each row
        represents a pixel and each column represents a band or channel. The pixel values
        are then scaled to the range [0, 1] using MinMaxScaler.
        Args:
            image_dataset (numpy.ndarray): The input image dataset. It can be either:
                - A 3D array (height x width x bands).
                - A 2D array (height x width).
        Returns:
            numpy.ndarray: A 2D array where each row corresponds to a pixel and each column
            corresponds to a band or channel. The values are scaled to the range [0, 1].
        Notes:
            - If the input is a 3D array, the output shape will be
              (height * width, bands).
            - If the input is a 2D array, the output shape will be
              (height * width, 1).
            - If the maximum value in the dataset is 0, scaling is skipped.
        """
        if image_dataset.ndim == 3:
            new_shape = (
                image_dataset.shape[2] * image_dataset.shape[1],
                image_dataset.shape[0],
            )
            img_as_2d_arr = image_dataset[:, :, :].reshape(new_shape)
        elif image_dataset.ndim == 2:
            new_shape = (image_dataset.shape[0] * image_dataset.shape[1], 1)
            img_as_2d_arr = image_dataset[:, :].reshape(new_shape)

        scaler = MinMaxScaler()
        if np.max(img_as_2d_arr) != 0:
            img_as_2d_arr = scaler.fit_transform(img_as_2d_arr)

        return img_as_2d_arr


class CloudDetectionModel:
    """
    CloudDetectionModel is a class designed to handle cloud detection in images using
    pre-trained machine learning models. It supports two types of models: logistic regression
    and naive Bayes.
    Attributes:
        model_type (str): The type of model to use ("logistic regression" or "naive bayes").
        model (object): The loaded machine learning model.
    Methods:
        __init__(model_type="logistic regression"):
            Initializes the CloudDetectionModel with the specified model type and loads
            the corresponding pre-trained model.
        predict(image):
            Predicts the presence of clouds in the given image and returns a 2D prediction map.
                image (object): An object containing image data and preprocessed image data.
            Returns:
                numpy.ndarray: A 2D array representing the prediction map, where each value
                indicates the presence or absence of clouds.
    """

    def __init__(self, model_type="logistic regression"):
        """
        Initializes the CloudDetectionModel class with the specified model type.
        Parameters:
            model_type (str): The type of model to use for cloud detection.
                              Options are "logistic regression" for logistic regression
                              or "naive bayes" for naive Bayes. Default is "logistic regression".
        Attributes:
            model_type (str): Stores the type of model specified.
            model (object): The loaded machine learning model based on the specified type.
        """

        self.model_type = model_type
        if model_type == "logistic regression":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "ml_models/logistic_regression.joblib")
            self.model = load(model_path)
        elif model_type == "naive bayes":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "ml_models/naive_bayes.joblib")
            self.model = load(model_path)

    def predict(self, image):
        """
        Predicts a cloud detection map for the given image.
        Args:
            image (object): An object containing image data and preprocessed image data.
                - image.image_data: A 3D array representing the original image data with dimensions
                  (channels, height, width).
                - image.image_data_preprocessed: A 2D array representing the preprocessed image data
                  suitable for model prediction.
        Returns:
            numpy.ndarray: A 2D array (height x width) representing the predicted cloud detection map.
        """

        height, width = image.image_data.shape[1], image.image_data.shape[2]
        image_2d = image.image_data_preprocessed.values
        predictions = self.model.predict(image_2d)
        prediction_map = predictions.reshape(height, width)

        return prediction_map


def detect_clouds(image_path, output_path, model_type="logistic regression"):
    """
    Detects clouds in a satellite image and saves the resulting prediction map.
    Args:
        image_path (str): The file path to the input satellite image.
        output_path (str): The file path to save the output prediction map.
        model_type (str, optional): The type of cloud detection model to use.
            Defaults to "logistic regression".
    Returns:
        str: The file path to the saved prediction map.
    """

    image_processor = ImageDataset(image_path)
    image_processor.load_image()
    image_processor.normalize_channels()
    image_processor.compute_features()

    model = CloudDetectionModel(model_type=model_type)
    prediction_map = model.predict(image_processor)

    profile = image_processor.profile
    profile.update(dtype=rasterio.float32, count=1, compress="lzw")

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction_map.astype(np.float32), 1)

    return output_path

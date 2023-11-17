#### Forest Fire Detection

***

This project aims to provide a fast, real-time prediction for forest fires from video cameras.

***

**Running the Program**

- To run the program with a live video, simply execute `main.py`.
  - This video is sourced from: http://66.119.104.155/mjpg/video.mjpg
- To run the program with the sample video, replace the `video_path` in line 18 of `main.py` with the path to the included `example_video.webm`.

***

**Image Dataset Sources**

The model was trained using the following image datasets:

- **FIRE Dataset**: https://www.kaggle.com/datasets/phylake1337/fire-dataset?resource=download
- **Effective Forest Fire Detection Dataset:** https://data.mendeley.com/datasets/g5nzp6j3bt/2

***

**Project Files**

- `README.md`:
  - This file.
- `main.py`
  - Contains the GUI and driver code for the application.
- `MotionDetection.py`:
  - The motion detection algorithm and model prediction code.
- `WaveletFeatureExtraction.py`:
  - The wavelet feature extractor code that takes a given image or video frame and extracts relevant features from the color histograms.
- `DatasetGeneration.py`:
  - Code for generating the `WAV` `.csv` files from the image datasets.
- `XGBoost_Model.ipynb`:
  - A Jupyter Notebook containing the full training and hyperparameter tuning process for the XGBoost model used in the motion detection code.
- `tuned_model.json`:
  - The fully trained and tuned XGBoost model used in the project.
- `WAV__effective_forest_fire_detection.csv`:
  - Wavelet features extracted from the Effective Forest Fire Detection image dataset.
- `WAV_fire_dataset.csv`:
  - Wavelet features extracted from the FIRE image dataset.
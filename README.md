
# Real-Time Face Mask and Social Distancing Detector

This project uses deep learning and computer vision to monitor public spaces for adherence to COVID-19 safety protocols. It includes two main functionalities:

1.  **Face Mask Detection**: Identifies whether individuals in a video stream are wearing a face mask.
2.  **Social Distancing Monitoring**: Detects groups of people who are violating social distancing guidelines.

The system processes video streams in real-time to provide visual feedback and alerts.

## ✨ Features

  * **Real-Time Detection**: Analyzes live video streams from webcams or video files.
  * **High-Accuracy Face Mask Detection**: Utilizes a deep learning model fine-tuned on the MobileNetV2 architecture.
  * **Social Distancing Analysis**: Employs the YOLOv3 object detection model to identify people and calculates the distance between them.
  * **Visual Alerts**: Overlays bounding boxes on the video feed, color-coded to indicate mask status and social distancing violations.

## 🛠️ Tech Stack & Dependencies

  * Python 3.8+
  * TensorFlow / Keras
  * OpenCV
  * Scikit-learn
  * NumPy
  * SciPy
  * imutils

## ⚙️ Setup and Installation

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2\. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

pip install -r requirements.txt
```

*(Note: You will need to create a `requirements.txt` file containing the necessary packages. A good starting point would be: `tensorflow, opencv-python, scikit-learn, numpy, scipy, imutils`)*

### 3\. Download Pre-trained Models

This project requires pre-trained models for face detection and person detection.

  * **Face Detector (Caffe Model):**

      * Download the model files from a trusted source for the ResNet-10 SSD face detector.
      * Place them in a `models/` directory:
          * `models/deploy.prototxt`
          * `models/res10_300x300_ssd_iter_140000.caffemodel`

  * **Person Detector (YOLOv3):**

      * Download the YOLOv3 weights, config, and COCO names file from the official [YOLO website](https://pjreddie.com/darknet/yolo/).
      * Place them in a `yolo-coco/` directory:
          * `yolo-coco/yolov3.weights`
          * `yolo-coco/yolov3.cfg`
          * `yolo-coco/coco.names`

## 🚀 How to Run

### 1\. Train the Face Mask Detector (Optional)

If you have your own dataset of faces with and without masks, you can train your own classifier.

  * **Dataset Structure:**

    ```
    dataset/
    ├── with_mask/
    │   ├── image1.jpg
    │   └── ...
    └── without_mask/
        ├── image2.jpg
        └── ...
    ```

  * **Run the training script:**

    ```bash
    python train_mask_detector.py --dataset path/to/your/dataset
    ```

    This will generate a `classifier.model` file (your trained model) and a `plot.png` showing the training accuracy/loss curves.

### 2\. Run the Social Distancing Detector

This script focuses solely on monitoring the distance between people.

  * **To use your webcam:**
    ```bash
    python social_distance_detector.py
    ```
  * **To use a pre-recorded video file:**
    ```bash
    python social_distance_detector.py --input your_video.mp4 --output output/result.avi
    ```

### 3\. Run the Combined Mask & Social Distance Detector

This script performs both face mask detection and social distancing monitoring.

  * **To use your webcam:**
    ```bash
    python mask_detector.py
    ```
  * **To use a pre-recorded video file:**
    ```bash
    python mask_detector.py --input your_video.mp4
    ```
    *(Press 'q' to stop the video stream)*

## 📂 Project Structure

```
.
├── models/                    # Stores the Caffe face detector model
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── yolo-coco/                 # Stores the YOLOv3 person detector model
│   ├── yolov3.weights
│   ├── yolov3.cfg
│   └── coco.names
├── dataset/                   # (Optional) For training your own mask detector
│   ├── with_mask/
│   └── without_mask/
├── train_mask_detector.py     # Script to train the face mask classifier
├── social_distance_detector.py # Script for real-time social distancing monitoring
├── mask_detector.py           # Script combining both mask and social distance detection
├── requirements.txt           # Project dependencies
└── README.md
```

## 🤝 Contributing

Contributions are welcome\! Please feel free to submit a pull request or open an issue if you have suggestions for improvements.

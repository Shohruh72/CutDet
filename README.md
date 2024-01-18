# Video Cut Detection.

This Python project offers a sophisticated solution for detecting cuts in videos and segmenting these videos into smaller clips. It leverages two distinct methods for cut detection: **Deep Learning** and **Optical Flow** analysis. The Deep Learning approach uses a pre-trained ResNet50 model to analyze frame features, while the Optical Flow method relies on frame differences and edge detection algorithms.

## Features

- **Deep Learning Based Cut Detection**: Utilizes the ResNet50 model to detect significant changes in video frames, indicating potential cuts.
- **Optical Flow Based Cut Detection**: Employs frame difference, edge detection, and optical flow calculations to identify cuts.
- **Automatic Video Segmentation**: Segments videos at detected cut points, saving them as individual files.
- **Flexible Method Selection**: Users can choose between the Deep Learning and Optical Flow methods for cut detection.


Make sure to install them using `pip` or `conda` before running the script.

## Installation

```bash
$ conda env create -f requirements.yaml
```

## Usage
To use this tool, follow these steps:

1. Place your video file(s) in an input directory within the project folder.
2. Run the script using Python:
3. When prompted, select the detection method (OptFlow or DeepLearning).

The script will process the video and output the segments into an outputs directory, organized by the selected method.
         

## Code Structure
* **VideoCutter:** Class implementing the Deep Learning based approach using ResNet50 for feature extraction.
* **VideoCutDetector:** Class for the Optical Flow based method, using frame differences and edge detection.
* **video_cut_detection():** Entry point of the script, handling user input and orchestrating the video processing.

## Customization
Please adjust the thresholds and parameters in the class initializations to suit your specific requirements.

## Output
* Detected cuts are saved in the outputs directory under respective method-named folders.

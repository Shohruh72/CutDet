# Import necessary libraries
import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.cluster import KMeans


# Define the VideoCutter class...
class VideoCutter:
    def __init__(self):
        # Load a pre-trained ResNet50 model and set it to evaluation mode.
        self.model = models.resnet50(pretrained=True).eval()

        # Define a series of image transformations to preprocess video frames.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Method to extract features from a single frame using the ResNet model.
    def extract_features(self, frame):
        frame = self.transform(frame)
        frame = frame.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            features = self.model(frame)

        return features  # Return the extracted features.

    # Calculate the change in features between consecutive frames in a video.
    def calculate_feature_changes(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return []

        feature_changes = []
        ret, previous_frame = cap.read()
        if not ret:
            return []

        previous_features = self.extract_features(previous_frame)

        while True:
            ret, current_frame = cap.read()
            if not ret:
                break
            if current_frame is None:
                continue  # Skip empty frames
            current_features = self.extract_features(current_frame)
            # Calculate the L2 norm between current and previous features to measure change.
            feature_change = torch.norm(current_features - previous_features, p=2).item()
            feature_changes.append(feature_change)

            previous_features = current_features

        cap.release()
        return feature_changes  # Return the list of feature changes.

    # Find the optimal threshold to identify cuts based on feature changes using KMeans clustering.
    def find_optimal_threshold(self, feature_changes):
        feature_changes = np.array(feature_changes).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(feature_changes)
        centroids = kmeans.cluster_centers_

        return np.mean(centroids)  # Return the average of the centroids as the threshold.

    # Detect the cut points in a video based on the calculated threshold.
    def detect_cuts(self, video_path):
        feature_changes = self.calculate_feature_changes(video_path)
        threshold = self.find_optimal_threshold(feature_changes)
        print("Calculated threshold:", threshold)

        cut_points = []
        for i, change in enumerate(feature_changes):
            if change > threshold:
                cut_points.append(i + 1)  # Add 1 to offset frame index

        return cut_points  # Return the list of cut points.

    # Method to save the segments of the video as individual files based on the detected cuts.
    def save_cuts(self, video_path, cut_points, method_name):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return []

        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Add the end of the video as the last cut point.
        cut_points.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        current_cut_index = 0
        output_files = []

        output_dir = f'outputs/{method_name}/'
        os.makedirs(output_dir, exist_ok=True)

        for i, cut_point in enumerate(cut_points):
            output_file = os.path.join(output_dir, f"Cut{i + 1}.mp4")
            out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

            while current_cut_index < cut_point:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                current_cut_index += 1

            out.release()
            output_files.append(output_file)

            if i < len(cut_points) - 2:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cut_points[i + 1])
                current_cut_index = cut_points[i + 1]

        cap.release()
        return output_files  # Return the list of output file names.


# Define the VideoCutDetector class for detecting cuts in videos.
class VideoCutDetector:
    def __init__(self, frame_diff_threshold=10, edge_diff_threshold=20, flow_threshold=1.0):
        self.frame_diff_threshold = frame_diff_threshold
        self.edge_diff_threshold = edge_diff_threshold
        self.flow_threshold = flow_threshold

    # Method to calculate the absolute difference between two frames.
    def calculate_frame_difference(self, current_frame, previous_frame):
        # Calculate frame difference
        return cv2.absdiff(current_frame, previous_frame)

    # Apply the Canny edge detection algorithm to a frame.
    def apply_edge_detection(self, frame):
        # Apply Canny edge detection
        return cv2.Canny(frame, 100, 200)

    # Calculate optical flow between two frames using Farneback's method.
    def calculate_optical_flow(self, current_frame, previous_frame):
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    # Detect cuts in a video file based on the set thresholds.
    def detect_cuts(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return []

        cut_points = []
        ret, previous_frame = cap.read()
        if not ret:
            print("Error: Unable to read first frame.")
            return []

        frame_index = 1
        while True:
            ret, current_frame = cap.read()
            if not ret:
                break

            # Calculate differences between consecutive frames using previously defined methods.
            frame_diff = self.calculate_frame_difference(current_frame, previous_frame)
            edges_current = self.apply_edge_detection(current_frame)
            edges_previous = self.apply_edge_detection(previous_frame)
            edge_diff = self.calculate_frame_difference(edges_current, edges_previous)
            optical_flow = self.calculate_optical_flow(current_frame, previous_frame)
            flow_magnitude = np.mean(np.linalg.norm(optical_flow, axis=2))

            # Determine if a cut is detected based on the thresholds.
            if (np.mean(frame_diff) > self.frame_diff_threshold or
                np.mean(edge_diff) > self.edge_diff_threshold) and flow_magnitude < self.flow_threshold:
                cut_points.append(frame_index)

            # Prepare for the next iteration.
            previous_frame = current_frame
            frame_index += 1

        cap.release()
        return cut_points  # Return the list of detected cut points.

    # Method to save the detected cuts as individual video files.
    def save_cuts(self, video_path, cut_points, method_name):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return []

        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cut_points.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        current_cut_index = 0
        output_files = []

        output_dir = f'outputs/{method_name}/'
        os.makedirs(output_dir, exist_ok=True)

        # Iterate over each cut point to save video segments.
        for i, cut_point in enumerate(cut_points):
            output_file = os.path.join(output_dir, f"Cut{i + 1}.mp4")
            out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

            while current_cut_index < cut_point:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                current_cut_index += 1

            out.release()
            output_files.append(output_file)

            if i < len(cut_points) - 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cut_points[i + 1])
                current_cut_index = cut_points[i + 1]

        cap.release()
        return output_files  # Return the list of filenames for the cut segments.


def main():
    video_path = 'input/Clip1.mp4'
    method = input("Select the method ('OptFlow' or 'DeepLearning'): ")

    if method.lower() == 'deeplearning':
        detector = VideoCutter()
        cuts = detector.detect_cuts(video_path)
        cut_video_files = detector.save_cuts(video_path, cuts, "DeepLearning")
    elif method.lower() == 'optflow':
        detector = VideoCutDetector()
        cuts = detector.detect_cuts(video_path)
        cut_video_files = detector.save_cuts(video_path, cuts, "OptFlow")
    else:
        raise ValueError("Invalid method selected.")

    print("Detected cuts at frames:", cuts)
    print("Cut video files:", cut_video_files)


if __name__ == "__main__":
    main()

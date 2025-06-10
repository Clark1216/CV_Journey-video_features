import numpy as np
import cv2

# Load the .npy file
file_path = './output/i3d/20250515_20250515155254_20250515155304_155254_timestamps_ms.npy'

video_path = './sample/GYY_Buckle_Asembly_Video/20250515_20250515155254_20250515155304_155254.mp4'
label_paths = ''

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
else:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)

cap.release()

data = np.load(file_path)

# Print the shape (dimensions)
print("Shape:", data.shape)

# Optional: print data type and size
print("Data type:", data.dtype)
print("Total elements:", data.size)

if file_path.endswith('_ms.npy'):
    print("Ending point of each step is:")
    print(data.tolist())

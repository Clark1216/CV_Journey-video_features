import numpy as np
import cv2

# Load the .npy file
# file_path = './output/merged_npy_Stack25_Step5/20250515_20250515155254_20250515155304_155254.npy'
file_path = './output/merged_npy_long_Stack50_Step25/10.50.7.232_04_20250704163158274_2560x1440_25fps.npy'
# file_path = './output/i3d/20250515_20250515155254_20250515155304_155254_timestamps_ms.npy'
# file_path = './output/i3d_S75_S5/20250515_20250515155254_20250515155304_155254_flow.npy'
# video_path = './sample/GYY_Buckle_Asembly_Video/20250515_20250515155254_20250515155304_155254.mp4'
video_path = './sample/GYY_Buckle_Asembly_Video_Long/10.50.7.232_04_20250704163158274_2560x1440_25fps.mp4'
label_paths = ''

# Frame counting
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file.")
else:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)
cap.release()

# npy data view
data = np.load(file_path)
## Print the shape (dimensions)
print("Shape:", data.shape)
## Optional: print data type and size
print("Data type:", data.dtype)
print("Total elements:", data.size)
## If npy is about timestamp info, print out
if file_path.endswith('_ms.npy'):
    print("Ending point of each step is:")
    print(data.tolist())

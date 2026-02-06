import numpy as np
import cv2
import os
from collections import defaultdict

#Parameters
# --- Compute half_stack_ms properly ---
FPS = 25
STACK_SIZE_FRAMES = 50
half_stack_ms = (STACK_SIZE_FRAMES / FPS) * 1000 / 2




# Load the .npy file
org_npy_dir='./output/i3d_longvideo_S50_S25'
org_label_dir='./sample/GYY_Buckle_Asembly_Label_Long'
# file_path = './output/i3d/20250515_20250515155254_20250515155304_155254_timestamps_ms.npy'

video_path_list_txt = './video_path.txt'
# video_path = './sample/GYY_Buckle_Asembly_Video/20250515_20250515155254_20250515155304_155254.mp4'
label_paths = ''

# Output folder
output_dir = './output/merged_npy_long_Stack50_Step25'
os.makedirs(output_dir, exist_ok=True)

def timestamp_to_ms(ts: str) -> int:
    """
    Supports:
      - MM:SS:sSS
      - SS:sSS  (fallback)
    """
    parts = ts.strip().split(":")
    if len(parts) == 3:
        mm, ss, sSS = map(int, parts)
        return ((mm) * 60 + ss + sSS/60) * 1000
    elif len(parts) == 2:
        seconds, sub_seconds = map(int, parts)
        total_ms = (seconds + sub_seconds/60) * 1000
        return total_ms
    else:
        raise ValueError(f"Unsupported timestamp format: {ts}")

LABEL_MAP = {
    "ning": "Take_part",
    "zhuang": "Screw_and_install",
    "steady": "Idle",
}

def normalize_label(s: str) -> str:
    return (s.strip()
              .replace("“", "")
              .replace("”", "")
              .replace('"', "")
              .replace("'", ""))

def load_intervals(label_txt_path: str):
    intervals = []
    with open(label_txt_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{label_txt_path} line {line_no} malformed: {line}")

            start_ts, end_ts = parts[0], parts[1]
            raw_label = normalize_label(" ".join(parts[2:]))
            label = LABEL_MAP.get(raw_label, raw_label)  # fallback keep original

            start_ms = timestamp_to_ms(start_ts)
            end_ms = timestamp_to_ms(end_ts)
            if end_ms <= start_ms:
                raise ValueError(f"{label_txt_path} line {line_no} has end<=start: {line}")

            intervals.append((start_ms, end_ms, label))

    intervals.sort(key=lambda x: x[0])
    return intervals

def labels_for_times(times_ms, intervals, default_label="UNLABELED"):
    # interval rule: start <= t < end
    out = []
    j = 0
    n = len(intervals)

    for t in times_ms:
        while j < n and t >= intervals[j][1]:
            j += 1
        if j < n and intervals[j][0] <= t < intervals[j][1]:
            out.append(intervals[j][2])   # already mapped to English here
        else:
            out.append(default_label)
    return out


# Read all file paths
with open(video_path_list_txt, 'r') as f:
    file_paths = [line.strip() for line in f if line.strip().endswith('.mp4')]

# Create nested dict: groups[basename][suffix] = list of files
groups = defaultdict(lambda: defaultdict(list))

for path in file_paths:
    basename = os.path.splitext(os.path.basename(path))[0]
    # print("XXXX", basename)
    flow_path = os.path.join(org_npy_dir, f'{basename}_flow.npy')
    rgb_path = os.path.join(org_npy_dir, f'{basename}_rgb.npy')
    ending_ts_path = os.path.join(org_npy_dir, f'{basename}_timestamps_ms.npy')
    original_label_path = os.path.join(org_label_dir, f'{basename}.txt')
    # print("XX", flow_path)
    # print("XXX", rgb_path)

    groups[basename]['flow']=flow_path
    groups[basename]['rgb']=rgb_path
    groups[basename]['ts']=ending_ts_path
    groups[basename]['org_label']=original_label_path


# Process and save
for group_key, files in groups.items():
    arrays = []
    rgb_data = np.load(files['rgb'])

    arrays.append(rgb_data)
    flow_data = np.load(files['flow'])
    arrays.append(flow_data)

    ## Stack feaures
    concatenated = np.concatenate(arrays, axis=1)
    output_path = os.path.join(output_dir, f'{group_key}.npy')
    np.save(output_path, concatenated)
    print(f"Saved merged array for '{group_key}' to {output_path}")
    ## Extract cut-off timestamp

    intervals = load_intervals(files["org_label"])
    # print("XXXXXXXXXX", intervals)
    ending_timestamps = np.load(files["ts"])  # assumed ms, increasing
    # choose the time you want to label: center of the stack
    center_times = ending_timestamps - half_stack_ms

    frame_labels = labels_for_times(center_times, intervals, default_label="UNLABELED")

    output_txt = os.path.join(output_dir, f"{group_key}.txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        for lb in frame_labels:
            f.write(lb + "\n")
    print(f"Saved converted label for '{group_key}' to {output_txt}")

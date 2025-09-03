import numpy as np
import cv2
import os
from collections import defaultdict

#Parameters
half_stack_size = 25/25*1000/2 # in unit of ms
q_stack_size = 25/25*1000/4
lb1 = 'Pick_up_part'
lb2 = 'Screw_and_install'
lb3 = 'Wrap_up'

# Load the .npy file
org_npy_dir='./output/i3d_S25_S5'
org_label_dir='./sample/GYY_Buckle_Asembly_Label'
file_path = './output/i3d/20250515_20250515155254_20250515155304_155254_timestamps_ms.npy'

video_path_list_txt = './video_path.txt'
video_path = './sample/GYY_Buckle_Asembly_Video/20250515_20250515155254_20250515155304_155254.mp4'
label_paths = ''

# Output folder
output_dir = './output/merged_npy_Stack25_Step5'
os.makedirs(output_dir, exist_ok=True)

def timestamp_to_ms(ts):
    """Converts 'SS:sSS' to milliseconds."""
    parts = ts.split(":")
    parts = [int(p) for p in parts]

    if len(parts) == 2:
        seconds, sub_seconds = parts
        total_ms = (seconds + sub_seconds/60) * 1000
    else:
        raise ValueError(f"Unsupported timestamp format: {ts}")

    return total_ms


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
    cut_off = []
    with open(files['org_label'], 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    ### Extract the second timestamp of line 1 and the first timestamp of line 2
    if len(lines) >= 2:
        first_line_parts = lines[0].split()
        second_line_parts = lines[1].split()
        if len(first_line_parts) >= 2 and len(second_line_parts) >= 1:
            cut_off.append(first_line_parts[1])
            cut_off.append(second_line_parts[1]) # '07:27'
    else:
        print("Label file {} seems wrong, please check". format(group_key))
    print("Cut-off timestamps:", cut_off)

    ## Touch up labels
    output_txt = os.path.join(output_dir, f'{group_key}.txt')
    ending_timestamps = np.load(files['ts'])
    cutoff1 = timestamp_to_ms(cut_off[0])
    cutoff2 = timestamp_to_ms(cut_off[1])
    with open(output_txt, 'w') as f:
        for timestamp in ending_timestamps:
            if timestamp-half_stack_size < cutoff1:
                f.write(f"{lb1}\n")
            elif timestamp-half_stack_size < cutoff2:
                f.write(f"{lb2}\n")
            else:
                f.write(f"{lb3}\n")
    print(f"Saved converted label for '{group_key}' to {output_txt}")

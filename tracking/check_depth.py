import cv2
import numpy as np
import subprocess

def get_depth_frame(video_path, frame_idx, width, height):
    # 使用 ffmpeg 提取特定帧并保持 16-bit 格式
    command = [
        'ffmpeg', '-i', video_path,
        '-vf', f'select=eq(n\,{frame_idx})',
        '-f', 'rawvideo',
        '-pix_fmt', 'gray16le',
        '-'
    ]
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=width*height*2)
    raw_output = pipe.stdout.read(width * height * 2)
    depth = np.frombuffer(raw_output, dtype=np.uint16).reshape((height, width))
    return depth

# 使用示例
frame = get_depth_frame("/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/videos_0121/squeegee_collect_sand/20260122_squeegee_collect_sand_from_table_20/cam0_depth.mkv", 0, 1280, 720)


# # Read one frame of depth
# cap = cv2.VideoCapture("/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/videos_0115/20260115_pinkknife_slice_unpealed_banana_2/cam0_depth.mkv")
# ret, frame = cap.read()
# cap.release()

# Extract 16-bit depth
if len(frame.shape) == 3:
    depth = frame[:, :, 0].astype(np.uint16) + (frame[:, :, 1].astype(np.uint16) << 8)
else:
    depth = frame

# Check depth value distribution
valid_depth = depth[depth > 0]
unique_values = np.unique(valid_depth)
print(f"Valid depth range: {valid_depth.min()} - {valid_depth.max()} mm")
print(f"Number of unique depth values: {len(unique_values)}")
print(f"First 20 unique values: {unique_values[:20]}")

# Check if there are many repeated values
hist, bins = np.histogram(valid_depth, bins=100)
print(f"\nDepth histogram (first 10 bins):")
for i in range(10):
    print(f"  {bins[i]:.0f}-{bins[i+1]:.0f}mm: {hist[i]} pixels")
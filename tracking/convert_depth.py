import subprocess
import numpy as np

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
depth_frame = get_depth_frame("/home/ruoqu/crq_ws/robotool/HO-Cap-Annotation/data/videos_0121/squeegee_collect_sand/20260122_squeegee_collect_sand_from_table_20/cam0_depth.mkv", 0, 1280, 720)
print(f"深度范围: {depth_frame.min()} - {depth_frame.max()}")
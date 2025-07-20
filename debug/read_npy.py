import numpy as np

def inspect_npy_file(file_path, print_values=False, max_elements=20):
    data = np.load(file_path, allow_pickle=True)

    print(f"Inspecting NPY file: {file_path}")
    print(f"Type: {type(data)}")
    
    if isinstance(data, np.ndarray):
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        
        if print_values:
            print(f"Values (up to {max_elements} elements):\n{data.flat[:max_elements]}")
    
    else:
        print("Loaded object is not an ndarray. It might be a Python object (dict, list, etc).")
        if print_values:
            print(data)

# 示例用法
if __name__ == "__main__":
    file_path = "/viscam/projects/robotool/data/videos_0713_annotated/296e90bd_brush_nuts_0/tool_masks/cam00.mp4/0.npy"  # 替换为你的文件路径
    inspect_npy_file(file_path, print_values=True)

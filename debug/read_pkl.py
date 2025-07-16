import pickle
import numpy as np

def print_list_shapes(sub_key, sub_value, level=1):
    """
    递归地打印 list 的长度（形状），不展示具体内容。
    """
    print(f"{'  ' * (level-1)}Sub-key: {sub_key}, List Level: {level}")
    
    if isinstance(sub_value, list):
        print(f"{'  ' * level}Length: {len(sub_value)}")
        for idx, item in enumerate(sub_value):
            # 如果 item 是 list 或 numpy 数组，递归打印其形状
            if isinstance(item, list) or isinstance(item, np.ndarray):
                print_list_shapes(f"Item {idx}", item, level + 1)
    elif isinstance(sub_value, np.ndarray):
        print(f"{'  ' * level}Array Shape: {sub_value.shape}")
    else:
        print(f"{'  ' * level}Value Type: {type(sub_value)}")


def load_pkl_and_print_hand_pose_info(pkl_file):
    # 加载 .pkl 文件
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # 打印字典的键和它们的对应值的类型
    print("Loaded data from .pkl file:")
    
    if isinstance(data, dict):
        print(f"Data is a dictionary with {len(data)} keys.")
        for key, value in data.items():
            print(f"Key: {key}, Value is of type {type(value)}")
    else:
        print(f"Data is not a dictionary. It is of type {type(data)}")

    # 提取 hand_pose 数据
    if 'hand_pose' in data:
        hand_pose = data['hand_pose']
        
        # 如果 hand_pose 是字典，查看里面的内容
        if isinstance(hand_pose, dict):
            print("hand_pose is a dictionary.")
            # 打印每个子项的长度、形状等信息
            for sub_key, sub_value in hand_pose.items():
                print(f"Sub-key: {sub_key}, Sub-value type: {type(sub_value)}")
                
                # 如果是 NumPy 数组，打印形状
                if isinstance(sub_value, np.ndarray):
                    print(f"Shape of {sub_key}: {sub_value.shape}")
                elif isinstance(sub_value, list):
                    # 如果是 list，递归打印每个维度的形状
                    print_list_shapes(sub_key, sub_value)
                else:
                    print(f"Unknown type for {sub_key}: {type(sub_value)}")
        else:
            print(f"hand_pose is not a dictionary. It is of type {type(hand_pose)}")
    else:
        print("No 'hand_pose' found in the data.")


# 示例：使用文件路径
pkl_file_path = '/home/wys/learning-compliant/crq_ws/robotool/HandReconstruction/coffee/2379b837_coffee_1/result_hand_optimized.pkl'  # 替换为你的 .pkl 文件路径

load_pkl_and_print_hand_pose_info(pkl_file_path)

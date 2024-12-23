import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class VideoDataset(Dataset):
    def __init__(self, data_dir, label_dir, t=10, transform=None):
        """
        Args:
            data_dir (string): 视频数据集的目录路径，其中包括每个视频的帧图片。
            label_dir (string): 标签目录路径，包含CSV标签文件。
            t (int): 每个视频中选择帧的间隔，例如每隔t帧选择1帧。
            transform (callable, optional): 可选的图像转换函数。
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.t = t  # 帧的间隔
        self.transform = transform
        self.videos = sorted(os.listdir(data_dir))  # 获取视频的文件夹名
        self.labels = self._load_labels(label_dir)  # 加载标签

    def _load_labels(self, label_dir):import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, t=10, transform=None):
        """
        :param data_dir: 数据集根目录，包含视频帧
        :param label_dir: 标签文件夹，包含 CSV 标签文件
        :param t: 每个视频选取的帧数
        :param transform: 数据预处理方法
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.t = t  # 每个视频选取的帧数
        self.transform = transform

        # 获取所有视频的文件夹
        self.video_folders = sorted(os.listdir(data_dir))

        # 读取标签文件
        self.label_files = sorted(os.listdir(label_dir))  # 获取标签文件列表
        self.labels = self._load_labels()  # 加载标签

    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, index):
        video_folder = self.video_folders[index]
        video_path = os.path.join(self.data_dir, video_folder, 'data')  # 视频数据文件夹

        # 获取视频帧路径
        frame_paths = sorted(os.listdir(video_path))  # 获取当前视频文件夹中的所有帧

        num_frames = len(frame_paths)  # 视频的总帧数
        frame_indices = torch.linspace(0, num_frames - 1, self.t).long()  # 计算均匀分布的索引

        selected_frames = []
        for idx in frame_indices:
            frame_path = os.path.join(video_path, frame_paths[idx.item()])
            image = Image.open(frame_path)  # 加载图片
            if self.transform:
                image = self.transform(image)  # 对图像进行预处理

            selected_frames.append(image)

        # 将选中的帧按张量堆叠
        video_frames = torch.stack(selected_frames)  # 形状： [t, C, H, W]

        # 获取视频的标签
        label = self.labels[video_folder]  # 使用文件夹名作为索引查找标签

        return video_frames, label

    def _load_labels(self):
        """
        加载所有标签，假设标签存储在CSV文件中
        :return: 返回一个字典，键为视频文件夹名，值为对应的视频标签
        """
        labels = {}

        # 遍历标签文件夹中的所有CSV文件
        for label_file in self.label_files:
            label_file_path = os.path.join(self.label_dir, label_file)

            # 读取CSV文件
            label_data = pd.read_csv(label_file_path)

            # 假设CSV文件中的每一行都包含一个视频的标签，并且视频文件夹名作为第一列
            for _, row in label_data.iterrows():
                video_folder_name = row[0]  # 第一列为视频文件夹名
                video_label = int(row[1])  # 第二列为视频标签（假设标签是整数）

                # 将标签存储在字典中，键为视频文件夹名
                labels[video_folder_name] = video_label

        return labels

        # 假设标签文件是CSV格式，读取所有CSV文件
        labels = {}
        for label_file in os.listdir(label_dir):
            if label_file.endswith('.csv'):
                df = pd.read_csv(os.path.join(label_dir, label_file))
                for idx, row in df.iterrows():
                    labels[row['frame_id']] = row['label']  # 假设CSV中有frame_id和label列
        return labels

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_dir = os.path.join(self.data_dir, self.videos[idx])  # 每个视频文件夹的路径
        frames = sorted(os.listdir(video_dir))  # 获取视频帧文件名，假设已经按帧编号排序

        # 选择每隔t帧取一个帧
        selected_frames = frames[::self.t]  # 每隔t帧选一帧

        images = []
        for frame in selected_frames:
            frame_path = os.path.join(video_dir, frame)
            image = Image.open(frame_path)  # 打开图片

            if self.transform:
                image = self.transform(image)

            images.append(image)

        # 假设标签是根据视频的ID或帧ID来分配的，选择当前视频的标签
        # 这里我们只是简单选择了第一个frame的标签，具体情况可以根据需求调整
        label = self.labels.get(frames[0], -1)  # 默认标签为-1，若找不到

        return torch.stack(images), label

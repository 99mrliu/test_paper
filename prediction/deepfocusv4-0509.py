import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os


# 定义 Inception 模块（1D 版本）
class Inception1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # 分支 1: 1x1 卷积
        self.branch1 = nn.Conv1d(in_channels, 32, kernel_size=1)

        # 分支 2: 1x1 -> 3x3
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1),
            nn.Conv1d(32, 32, kernel_size=3, padding=1)
        )

        # 分支 3: 1x1 -> 5x5
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1),
            nn.Conv1d(32, 32, kernel_size=5, padding=2)
        )

        # 分支 4: 最大池化 -> 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, 32, kernel_size=1)
        )

    def forward(self, x):
        b1 = F.relu(self.branch1(x))
        b2 = F.relu(self.branch2(x))
        b3 = F.relu(self.branch3(x))
        b4 = F.relu(self.branch4(x))
        return torch.cat([b1, b2, b3, b4], dim=1)


class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.mid_channels = channels // reduction

        # 序列方向注意力
        self.conv_seq = nn.Sequential(
            nn.Conv1d(1, self.mid_channels, 1),  # 修改 in_channels 为 1
            nn.ReLU(),
            nn.Conv1d(self.mid_channels, channels, 1),
            nn.Sigmoid()
        )

        # 特征方向注意力
        self.conv_feat = nn.Sequential(
            nn.Conv1d(channels, self.mid_channels, 1),
            nn.ReLU(),
            nn.Conv1d(self.mid_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入形状: (B, C, T)
        B, C, T = x.size()

        # 序列方向注意力
        x_seq = x.mean(1, keepdim=True)  # (B, 1, T)
        x_seq = self.conv_seq(x_seq)  # (B, channels, T)

        # 特征方向注意力
        x_feat = x.mean(2, keepdim=True)  # (B, C, 1)
        x_feat = self.conv_feat(x_feat)  # (B, 1, 1)
        x_feat = x_feat.expand(-1, -1, T)  # (B, 1, T)

        # 合并注意力
        attention = torch.sigmoid(x_seq + x_feat)
        return x * attention


# 定义 Transformer 编码块（使用 CA 注意力）
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = CoordinateAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 输入形状调整: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)

        # 自注意力
        attn_out = self.attention(x.permute(0, 2, 1))  # (B, C, T)
        attn_out = attn_out.permute(0, 2, 1)  # (B, T, C)
        x = self.norm1(x + attn_out)

        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x.permute(0, 2, 1)  # 恢复为 (B, C, T)


# 定义完整模型
class CustomModel(nn.Module):
    def __init__(self, input_shape=(23, 9)):
        super().__init__()
        self.in_channels = input_shape[1]  # 输入特征维度为 9

        # Inception 特征提取
        self.inception = nn.Sequential(
            Inception1D(self.in_channels),
            Inception1D(128)  # 第一个 Inception 输出通道为 4*32=128
        )

        # Transformer 编码层
        self.transformer = nn.Sequential(
            TransformerBlock(128),
            TransformerBlock(128)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入形状: (B, T, C) = (B, 23, 9)
        x = x.permute(0, 2, 1)  # 转换为 (B, C, T) = (B, 9, 23)

        # Inception 特征提取
        x = self.inception(x)  # (B, 128, 23)

        # Transformer 处理
        x = self.transformer(x)  # (B, 128, 23)

        # 分类
        return self.classifier(x)


# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, npz_file):
        # 加载数据
        data = np.load(npz_file)
        self.features = data['data'].astype(np.float32)  # (122673, 23, 9)
        self.labels = data['labels'].astype(np.float32)  # (122673,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 早期停止类，用于保存最佳模型
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 训练函数
def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001,
                save_path='best_model.pth'):
    # 创建保存模型的目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 损失函数和优化器
    criterion = torch.nn.BCELoss()  # 二分类问题
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化早期停止
    early_stopping = EarlyStopping(patience=3, verbose=True, path=save_path)

    # 训练
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output.squeeze(), target)
                val_loss += loss.item()
                val_preds.extend(output.squeeze().round().cpu().numpy())
                val_labels.extend(target.cpu().numpy())

        val_accuracy = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Val Loss: {val_loss / len(val_loader):.4f}, "
              f"Val Accuracy: {val_accuracy * 100:.2f}%")

        # 检查是否需要早停
        early_stopping(val_loss / len(val_loader), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 加载最佳模型
    model.load_state_dict(torch.load(save_path))
    return model


# 评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            preds.extend(output.squeeze().round().cpu().numpy())
            labels.extend(target.cpu().numpy())

    accuracy = accuracy_score(labels, preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


# 预测函数
def predict(model, test_data, device):
    model.eval()
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(test_data)
        preds = output.squeeze().round().cpu().numpy()
    return preds


# 加载模型函数
def load_model(model_path, input_shape=(23, 9)):
    model = CustomModel(input_shape)
    model.load_state_dict(torch.load(model_path))
    return model


# 示例用法
if __name__ == "__main__":
    npz_file = ''
    dataset = CustomDataset(npz_file)

    # 创建数据加载器
    batch_size = 32
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomModel().to(device)

    # 模型保存路径-
    save_path = ''

    # 训练模型
    model = train_model(model, train_loader, val_loader, device,
                        num_epochs=15, learning_rate=0.001, save_path=save_path)

    # 评估模型
    evaluate_model(model, val_loader, device)  # 使用验证集评估

    # 预测新数据
    # test_data = ...  # 您的测试数据，形状为 (N, 23, 9)
    # preds = predict(model, test_data, device)

    # 加载保存的模型示例
    # loaded_model = load_model(save_path).to(device)
    # evaluate_model(loaded_model, val_loader, device)
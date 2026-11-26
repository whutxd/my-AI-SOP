import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ===================== 你的配置 =====================
SEQUENCE_LEN = 48    # 时序长度
FEATURE_DIM = 131    # 你定义的维度：5 + 2*21*3
NUM_CLASSES = 3      # 0=违规, 1=合规
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = np.load("lstm_runs_fine/data/npy/x.npy")
y = np.load("lstm_runs_fine/data/npy/y.npy")

# ====================================================

# 1. 数据集加载类
class SOPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    
class LSTMClassifier(nn.Module):
    """
    基于LSTM的动作分类模型
    输入序列特征，输出动作类别概率
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: [batch, seq_len, input_size]
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出进行分类
        return self.fc(out[:, -1, :])
    

# 3. 加载数据

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# 4. DataLoader
train_dataset = SOPDataset(X_train, y_train)
test_dataset = SOPDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. 初始化模型
model = LSTMClassifier(
    input_size=FEATURE_DIM,
    hidden_size=128,
    num_layers=2,
    num_classes=NUM_CLASSES
).to(DEVICE)

# 6. 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 7. 训练循环
print(f"训练设备: {DEVICE}")
print(f"数据形状: {X.shape}")

best_test_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    pred_train, true_train = [], []

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        pred_train.extend(pred.cpu().numpy())
        true_train.extend(batch_y.cpu().numpy())

    train_acc = accuracy_score(true_train, pred_train)
    avg_loss = train_loss / len(train_loader)

    # 测试
    model.eval()
    pred_test, true_test = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            outputs = model(batch_X)
            pred = torch.argmax(outputs, dim=1)
            pred_test.extend(pred.cpu().numpy())
            true_test.extend(batch_y.cpu().numpy())

    test_acc = accuracy_score(true_test, pred_test)
    print(f"Epoch [{epoch+1:2d}/{EPOCHS}] | "
          f"Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Test Acc: {test_acc:.4f}")

    # ===================== 核心修改：保存最优权重 =====================
    if test_acc > best_test_acc:
        # 更新最佳准确率
        best_test_acc = test_acc
        # 保存测试集acc最高的模型权重
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ 最佳模型已更新！测试准确率: {best_test_acc:.4f}")
    # =================================================================

# 训练全部结束后：保存最终权重
torch.save(model.state_dict(), "final_model.pth")
print("\n🏁 训练完成！")
print(f"📌 最佳测试准确率: {best_test_acc:.4f}")
print("💾 已保存文件：")

# 9. 最终评估
print("\n===== 测试集分类报告 =====")
print(classification_report(true_test, pred_test, target_names=["_background","get_cup", "drop_cup"]))
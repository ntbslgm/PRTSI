import matplotlib.pyplot as plt

def read_loss_data_1(file_path):
    # 初始化列表
    epochs = []
    cls_loss = []
    making_loss = []

    with open(file_path, 'r') as file:
        for line in file:
            if "Epoch" in line:
                epoch = int(line.split(':')[1].strip().split('/')[0])
                epochs.append(epoch)
            elif "cls_loss" in line:
                loss = float(line.split(':')[1].strip())
                cls_loss.append(loss)
            elif "making_loss" in line:
                loss = float(line.split(':')[1].strip())
                making_loss.append(loss)

    return epochs, cls_loss, making_loss

def read_loss_data_2(file_path):
    # 初始化列表
    epochs = []
    entropy_loss = []
    masking_loss = []

    with open(file_path, 'r') as file:
        for line in file:
            if "Epoch" in line:
                epoch = int(line.split(':')[1].strip().split('/')[0])
                epochs.append(epoch)
            elif "entropy_loss" in line:
                loss = float(line.split(':')[1].strip())
                entropy_loss.append(loss)
            elif "Masking_loss" in line:
                loss = float(line.split(':')[1].strip())
                masking_loss.append(loss)

    return epochs, entropy_loss, masking_loss

# 读取数据
epochs_1, cls_loss_1, making_loss_1 = read_loss_data_1('EEG-0-11.txt')
epochs_2, entropy_loss_2, masking_loss_2 = read_loss_data_2('EEG-0-11-Adaptation.txt')

# 绘图
plt.figure(figsize=(12, 6))

# 绘制第一个文件的数据
plt.plot(epochs_1, cls_loss_1, label='cls_loss (Pretraining)', color='blue', marker='o')
plt.plot(epochs_1, making_loss_1, label='making_loss (Pretraining)', color='orange', marker='x')

# 绘制第二个文件的数据
plt.plot(epochs_2, entropy_loss_2, label='entropy_loss (adaptation)', color='green', marker='s')
plt.plot(epochs_2, masking_loss_2, label='making_loss (adaptation)', color='red', marker='d')

# 设置图形属性
plt.title('Loss Over Epochs from Two Files')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
# plt.xticks(range(0, 101, 10))
plt.xticks(range(0, 41, 10))
plt.tight_layout()
plt.show()
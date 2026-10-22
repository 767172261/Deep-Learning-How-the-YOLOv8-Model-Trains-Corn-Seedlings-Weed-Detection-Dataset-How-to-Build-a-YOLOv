深度学习YOLOv8模型如何训练玉米幼苗杂草检测数据集  如何构建基于 YOLOv8 的玉米幼苗与杂草识别系统
@[toc]
以下文字及代码仅供参考学习使用。
## 数据集描述：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/aee0080106454fa5a128b33f5eaf2a02.png)

玉米幼苗杂草数据集
共3042张，已划分数据集

玉米幼苗和杂草两类，YOLO标签标注。训练集2433张，验证集609张。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0462ea727a134120b776aebf5b2e2477.png)

### 🌽 玉米幼苗和杂草检测数据集

#### 基础信息

| 项目             | 内容                    |
|------------------|-------------------------|
| 总图像数量       | 3,042 张                |
| 训练集（train）  | 2,433 张（80%）         |
| 验证集（val）    | 609 张（20%）           |
| 类别总数         | 2 类                    |

---

#### 类别详情统计表

| 序号 | 中文名称   | 英文类别名 | 图片数  | 标注框数  |
|------|------------|------------|---------|-----------|
| 1    | 玉米幼苗   | corn       | 3,042   | 12,156    |
| 2    | 杂草       | weed       | 2,594   | 11,168    |

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7c4d036af07c40c997b3dcfeb6898e9e.png)
# 如何构建基于 YOLOv8 的玉米幼苗与杂草识别系统

---

## 🧰 环境搭建（CUDA驱动、Anaconda、Python环境）

### 1. 安装 CUDA 驱动
```bash
nvidia-smi
```
- 如果未安装，请前往 [NVIDIA官网](https://www.nvidia.com/Download/index.aspx) 下载对应显卡型号的最新驱动。

---

### 2. 安装 Anaconda

从 [Anaconda官网](https://www.anaconda.com/products/distribution) 下载并安装适合你系统的版本。

---

### 3. 创建 Python 虚拟环境

```bash
conda create --name yolov8_weed python=3.9
conda activate yolov8_weed
```

---

### 4. 安装依赖项

```bash
pip install torch torchvision torchaudio
pip install ultralytics==8.2.0
pip install opencv-python
pip install numpy
pip install matplotlib
pip install tqdm
```

---

## 📁 数据集结构说明

假设同学你的数据集应组织如下：

```
corn_weed_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

### `data.yaml` 内容如下：

```yaml
train: ./corn_weed_dataset/images/train
val: ./corn_weed_dataset/images/val

nc: 2
names: ['corn', 'weed']
```

---

## 🏋️ 使用 YOLOv8 训练模型

### 加载官方预训练模型进行训练

```python
from ultralytics import YOLO

# 加载YOLOv8预训练模型（如 yolov8s.pt）
model = YOLO('yolov8s.pt')  # 可选：yolov8n, yolov8m, yolov8l, yolov8x
```

### 开始训练（推荐配置）

```python
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='corn_weed_yolov8s',
    pretrained=True,
    optimizer='AdamW',
    lr0=1e-3,
    lrf=1e-4,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=10,
    device=0 if torch.cuda.is_available() else None,
    workers=4,
    project="runs/corn_weed",
)
```

---

## 🔍 推理代码（单张图片 + 批量处理）

### 单张图片推理

```python
from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO('runs/corn_weed/corn_weed_yolov8s/weights/best.pt')

# 图片路径
image_path = 'test_images/example.jpg'

# 进行预测
results = model(image_path)

# 绘制结果
for result in results:
    annotated_img = result.plot()
    cv2.imshow('Detection Result', annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

---

### 批量图片推理

```python
import os
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/corn_weed/corn_weed_yolov8s/weights/best.pt')

# 输入输出目录
input_dir = 'corn_weed_dataset/images/test'
output_dir = 'inference_results'

os.makedirs(output_dir, exist_ok=True)

# 批量处理
for img_file in os.listdir(input_dir):
    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, img_file)
        results = model(image_path)
        for result in results:
            annotated_img = result.plot()
            output_path = os.path.join(output_dir, img_file)
            cv2.imwrite(output_path, annotated_img)

print(f"✅ 批量推理完成，结果已保存到 {output_dir}")
```

---

## 📊 模型评估代码（验证精度、混淆矩阵等）

### 验证 mAP、Recall、Precision 等指标

```python
# 验证模型性能
metrics = model.val()

# 输出各项指标
print("mAP@0.5:", metrics.box.map50)
print("mAP@0.5:0.95:", metrics.box.map)
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
print("F1 Score:", metrics.box.f1)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7ff9148c5ff94b79afde5d06e8c56845.png)

### 可视化混淆矩阵（需额外代码）

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 获取混淆矩阵
confusion_matrix = metrics.confusion_matrix

plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['corn', 'weed'], yticklabels=['corn', 'weed'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
```

---

## 🖥️ 实时视频检测系统（Tkinter GUI 示例）

```python
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class CornWeedDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("玉米与杂草实时检测 - YOLOv8")

        self.model = YOLO('runs/corn_weed/corn_weed_yolov8s/weights/best.pt')
        self.cap = cv2.VideoCapture(0)

        self.panel = tk.Label(root)
        self.panel.pack(pady=10)

        self.count_label = tk.Label(root, text="玉米: 0 | 杂草: 0")
        self.count_label.pack()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame)
            
            corn_count = 0
            weed_count = 0
            
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    
                    if conf > 0.5:
                        r = box.xyxy
                        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                        cv2.rectangle(frame, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, 2)
                        
                        label = f'{self.model.names[cls]} {conf:.2f}'
                        cv2.putText(frame, label, (int(r[0]), int(r[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        if cls == 0:
                            corn_count += 1
                        else:
                            weed_count += 1
                
                self.count_label.config(text=f"玉米: {corn_count} | 杂草: {weed_count}")

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize((640, 480))
            imgtk = ImageTk.PhotoImage(image=img)
            self.panel.imgtk = imgtk
            self.panel.configure(image=imgtk)
        
        self.root.after(10, self.update_frame)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = CornWeedDetector(root)
    root.mainloop()
```

---



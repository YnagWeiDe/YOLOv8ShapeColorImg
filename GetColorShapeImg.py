import os
import random
import math
from pathlib import Path
from PIL import Image, ImageDraw

# ========== 配置 ==========

IMG_WIDTH = 1080             # 图片宽度
IMG_HEIGHT = 720             # 图片高度
NUM_IMAGES = 2000            # 图片数量

# YOLOv8标准目录结构
DATASET_DIR = "dataset\\bvn"
IMG_DIR = os.path.join(DATASET_DIR, "images")
LBL_DIR = os.path.join(DATASET_DIR, "labels")
TXT_DIR = "txt"              # 用于存放classes.txt和全量txt
SPLITS = ["train", "val"]
VAL_RATIO = 0.2

# ========== 颜色定义（标签层面）==========
# 红橙黄绿青蓝紫 + 黑 + 白
BASE_COLORS = {
    "red":    (255, 0, 0),
    "orange": (255, 165, 0),
    "yellow": (255, 255, 0),
    "green":  (0, 255, 0),
    "cyan":   (0, 255, 255),
    "blue":   (0, 0, 255),
    "purple": (128, 0, 128),
    "black":  (0, 0, 0),
    "white":  (255, 255, 255),
}

# 每种颜色允许的抖动范围（RGB 每个通道上下浮动值）
# 可以根据需要调大或调小
COLOR_JITTER = {
    "red":    40,
    "orange": 40,
    "yellow": 40,
    "green":  40,
    "cyan":   40,
    "blue":   40,
    "purple": 40,
    "black":  25,   # 黑色在 0~25 之间波动
    "white":  25,   # 白色在 230~255 附近波动（会做裁剪）
}

# 形状类别
SHAPES = ["circle", "square", "rectangle", "triangle", "star", "diamond"]

# 颜色_形状组合类别（YOLO 的 class 映射）
COLORS = list(BASE_COLORS.keys())
COMBO_CLASSES = [f"{color}_{shape}" for color in COLORS for shape in SHAPES]

# ========== 创建目录结构 ==========
for split in SPLITS:
    Path(os.path.join(IMG_DIR, split)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(LBL_DIR, split)).mkdir(parents=True, exist_ok=True)
Path(TXT_DIR).mkdir(exist_ok=True)

# ========== 工具函数 ==========

def clamp(v, lo=0, hi=255):
    return max(lo, min(hi, v))

def sample_color_from_base(color_name):
    """
    根据基础颜色，在一定范围内随机抖动，返回用于绘制的 RGB。
    标签仍然使用 color_name，不受抖动影响。
    """
    base_r, base_g, base_b = BASE_COLORS[color_name]
    jitter = COLOR_JITTER.get(color_name, 30)

    r = clamp(random.randint(base_r - jitter, base_r + jitter))
    g = clamp(random.randint(base_g - jitter, base_g + jitter))
    b = clamp(random.randint(base_b - jitter, base_b + jitter))

    # 特别照顾一下白色，尽量保证是比较亮的白
    if color_name == "white":
        r = max(r, 230)
        g = max(g, 230)
        b = max(b, 230)

    return (r, g, b)

def get_bounding_box(pts):
    """根据一堆点计算最小外接矩形"""
    min_x = min(pts, key=lambda p: p[0])[0]
    max_x = max(pts, key=lambda p: p[0])[0]
    min_y = min(pts, key=lambda p: p[1])[1]
    max_y = max(pts, key=lambda p: p[1])[1]
    return (min_x, min_y, max_x, max_y)

# 各种形状的绘制函数
def draw_circle(draw, bbox, color):
    draw.ellipse(bbox, fill=color)
    return bbox

def draw_rectangle(draw, bbox, color):
    draw.rectangle(bbox, fill=color)
    return bbox

def draw_square(draw, bbox, color):
    x1, y1, x2, y2 = bbox
    side = min(x2 - x1, y2 - y1)  # 保证正方形
    new_bbox = (x1, y1, x1 + side, y1 + side)
    draw.rectangle(new_bbox, fill=color)
    return new_bbox

def draw_triangle(draw, bbox, color):
    x1, y1, x2, y2 = bbox
    p1 = ((x1 + x2) // 2, y1)
    p2 = (x1, y2)
    p3 = (x2, y2)
    draw.polygon([p1, p2, p3], fill=color)
    return bbox

def draw_star(draw, bbox, color):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r_outer = min(x2 - x1, y2 - y1) / 2
    r_inner = r_outer * 0.5
    pts = []
    for i in range(10):
        ang = i * math.pi / 5 - math.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    draw.polygon(pts, fill=color)
    return get_bounding_box(pts)

def draw_diamond(draw, bbox, color):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    points = [(cx, y1), (x2, cy), (cx, y2), (x1, cy)]
    draw.polygon(points, fill=color)
    return bbox

# 形状名到绘制函数的映射（方便后面调用）
DRAW_FUNCS = {
    "circle":    draw_circle,
    "square":    draw_square,
    "rectangle": draw_rectangle,
    "triangle":  draw_triangle,
    "star":      draw_star,
    "diamond":   draw_diamond,
}

# ========== 主程序 ==========

from random import shuffle
indices = list(range(NUM_IMAGES))
shuffle(indices)
split_idx = int(NUM_IMAGES * (1 - VAL_RATIO))
split_map = ["train" if idx < split_idx else "val" for idx in range(NUM_IMAGES)]

for i in range(NUM_IMAGES):
    # 白色图案使用黑色背景，其它颜色使用白色背景
    if random.choice([True, False]):  # 随机决定图案是否为白色
        bg_color = (0, 0, 0)  # 黑色背景
        color_name = "white"
    else:
        bg_color = (255, 255, 255)  # 白色背景
        color_name = random.choice(COLORS)
    
    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), bg_color)
    draw = ImageDraw.Draw(img)

    # 实际绘制用的颜色：在对应颜色范围内随机
    fill_color = sample_color_from_base(color_name)

    shape = random.choice(SHAPES)
    combo_name = f"{color_name}_{shape}"
    class_id = COMBO_CLASSES.index(combo_name)

    # 随机位置，保证不贴边
    margin = 50
    max_shape_w = IMG_WIDTH - 2 * margin
    max_shape_h = IMG_HEIGHT - 2 * margin
    shape_w = random.randint(int(max_shape_w * 0.3), int(max_shape_w * 0.7))
    shape_h = random.randint(int(max_shape_h * 0.3), int(max_shape_h * 0.7))
    x1 = random.randint(margin, IMG_WIDTH - margin - shape_w)
    y1 = random.randint(margin, IMG_HEIGHT - margin - shape_h)
    x2 = x1 + shape_w
    y2 = y1 + shape_h
    bbox = (x1, y1, x2, y2)

    # 绘制形状 & 拿到最终外接框（有的形状会调整 bbox）
    draw_func = DRAW_FUNCS[shape]
    bbox = draw_func(draw, bbox, fill_color)

    # 生成YOLOv8标注txt（归一化）
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / IMG_WIDTH
    y_center = ((y1 + y2) / 2) / IMG_HEIGHT
    w = (x2 - x1) / IMG_WIDTH
    h = (y2 - y1) / IMG_HEIGHT
    norm_text = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

    base_name = f"{shape}_{color_name}_{i+1:04d}"
    img_name = base_name + ".png"
    txt_name = base_name + ".txt"

    txt_path = os.path.join(TXT_DIR, txt_name)
    with open(txt_path, 'w') as f:
        f.write(norm_text + "\n")

    split = split_map[i]
    img_out_path = os.path.join(IMG_DIR, split, img_name)
    lbl_out_path = os.path.join(LBL_DIR, split, txt_name)
    img.save(img_out_path)
    with open(lbl_out_path, 'w') as f:
        f.write(norm_text + "\n")

    print(f"[{i+1}/{NUM_IMAGES}] saved: {img_out_path}  label: {lbl_out_path}")

# 生成classes.txt
classes_path = os.path.join(TXT_DIR, "classes.txt")
with open(classes_path, 'w', encoding='utf-8') as f:
    for cname in COMBO_CLASSES:
        f.write(cname + "\n")

print("✔ 数据集已按YOLOv8格式生成，图片和标注已分配到 dataset/images/train, dataset/images/val, dataset/labels/train, dataset/labels/val，类别文件为 txt/classes.txt")

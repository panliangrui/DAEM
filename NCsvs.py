import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import openslide
import cv2
import tifffile
import os
import matplotlib.patches as patches
# -------------------------------
# 1. 读取 CSV 数据及原始SVS图像参数
# -------------------------------
folder_path = './input'
file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

print(file_paths)
dpi=1000
# 打开 SVS 文件
svs_path = file_paths[0]  # 替换为实际的 SVS 文件路径
slide = openslide.OpenSlide(svs_path)

# 获取最高分辨率下的尺寸
width, height = slide.dimensions
print(f"SVS 图像尺寸：宽度={width}, 高度={height}")


# 假设存储在nuclei_data.csv中P:\\project_STAS\\STAS_2025\\TME_some\\oun
filename = [os.path.splitext(f)[0] for f in os.listdir('./input') if os.path.isfile(os.path.join('./input', f))]
filename = filename[0]
df = pd.read_csv('./output_all/{filename}.nuclei.csv'.format(filename=filename))
svs_path = f'./input/{filename}.svs'

# 使用 OpenSlide 读取原始SVS图像元数据
slide = openslide.OpenSlide(svs_path)
level_dims = slide.level_dimensions  # 比如 [(44400, 63104), (11100, 15776), (2775, 3944), (693, 986)]
properties = slide.properties
# 提取部分参数
app_mag = properties.get('aperio.AppMag', '20')
mpp_x = float(properties.get('openslide.mpp-x', '0.238204'))
mpp_y = float(properties.get('openslide.mpp-y', '0.238204'))

# -------------------------------
# 2. 统计细胞标签并计算STR
# -------------------------------
value_counts = df['labels'].value_counts()
for label, count in value_counts.items():
    if label == 1:
        print(f"tumor: {label}, Count: {count}")
        tumor = count
    if label == 2:
        print(f"stromal: {label}, Count: {count}")
        stromal = count
    if label == 3:
        print(f"immune: {label}, Count: {count}")
    if label == 4:
        print(f"blood: {label}, Count: {count}")
    if label == 5:
        print(f"macrophage: {label}, Count: {count}")
    if label == 6:
        print(f"dead: {label}, Count: {count}")
    if label == 7:
        print(f"other: {label}, Count: {count}")


# -------------------------------
# 3. 构建细胞空间图
# -------------------------------
# 提取中心点坐标 (x_c, y_c)
coordinates = df[['x_c', 'y_c']].values

# 构建无向图：节点代表细胞，其属性包含坐标和标签
G = nx.Graph()
for i in range(len(coordinates)):
    G.add_node(i, pos=(coordinates[i][0], coordinates[i][1]), label=df['labels'][i])

# 配置颜色映射
labels_color = {-100: '#949494', 0: '#ffffff', 1: '#ff0000', 2: '#00ff00', 3: '#0000ff',
                4: '#ff00ff', 5: '#ffff00', 6: '#0094e1', 7: '#646464'}

# 创建绘图对象，注意图大小可先用一个合适的尺寸绘制，然后后续统一调整到原始SVS尺寸
target_width, target_height = level_dims[0]
x = int(target_width/500)
y = int(target_height/500)
fig, ax = plt.subplots(figsize=(x, y))
pos = nx.get_node_attributes(G, 'pos')
labels = nx.get_node_attributes(G, 'label')
node_colors = [labels_color.get(labels[i], '#000000') for i in G.nodes()]
nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=30, ax=ax)


# 设置坐标轴显示范围以去除多余的空白
x_vals = [pos[i][0] for i in pos]
y_vals = [pos[i][1] for i in pos]
ax.set_xlim([min(x_vals), max(x_vals)])  # 根据数据动态设置 x 轴范围
ax.set_ylim([min(y_vals), max(y_vals)])  # 根据数据动态设置 y 轴范围

# 去掉坐标轴
ax.axis('off')


# for label, color in labels_color.items():
#     ax.scatter([], [], c=[color], label=label)
# ax.legend(scatterpoints=1, frameon=False, labelspacing=1, title="Cell Types")
# ax.set_title('WSI Nuclei Spatial Distribution')
ax.invert_yaxis()
# -------------------------------
# 4. 获取绘图图像数据（overlay图）
# -------------------------------
# 这里不直接保存，而是先获取图像数组
fig.canvas.draw()
overlay_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
overlay_img = overlay_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
plt.close(fig)

# -------------------------------
# 5. 调整overlay图像大小到原始SVS level 0尺寸
# -------------------------------
# 原始SVS level0尺寸 (宽度, 高度)
target_width, target_height = level_dims[0]
overlay_resized = cv2.resize(overlay_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

# -------------------------------
# 6. 构建金字塔：依照原始SVS多层尺寸生成下采样图像
# -------------------------------
pyramid = [overlay_resized]
for level in range(1, len(level_dims)):
    level_width, level_height = level_dims[level]
    level_img = cv2.resize(overlay_resized, (level_width, level_height), interpolation=cv2.INTER_AREA)
    pyramid.append(level_img)

# -------------------------------
# 7. 保存为SVS格式（多分辨率TIFF）
# -------------------------------
# SVS文件本质上是多分辨率的TIFF文件，需要写入基底图像及其下层。
# tifffile.imwrite支持subIFDs来写入金字塔
out_svs = f'./input\\{filename}_nuclei_spatial_distribution.svs'


tile_size = (512, 512)  # 设置瓦片大小
compression = 'jpeg'    # 设置压缩方式

with tifffile.TiffWriter(out_svs, bigtiff=True) as tiff:
    # 写入基底图像
    tiff.write(
        pyramid[0],
        photometric='rgb',
        tile=tile_size,
        compression=compression,
        metadata={
            'openslide.level-count': str(len(level_dims)),
            'openslide.level[0].width': str(level_dims[0][0]),
            'openslide.level[0].height': str(level_dims[0][1]),
            'openslide.mpp-x': str(mpp_x),
            'openslide.mpp-y': str(mpp_y),
            'aperio.AppMag': app_mag,
            'tiff.ResolutionUnit': 'inch'
        },
        subfiletype=0
    )
    # 写入金字塔下层作为 subIFDs
    for level_img in pyramid[1:]:
        tiff.write(
            level_img,
            photometric='rgb',
            tile=tile_size,
            compression=compression,
            subfiletype=1,
            contiguous=False
        )


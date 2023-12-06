import os
import shutil
import json
import random
import xml.etree.ElementTree as ET

def voc_to_coco_with_split(input_dir, output_dir, useful_labels, split_ratio=0.8):
    coco_data_train = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    coco_data_val = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    categories = {}
    category_id = 1

    # 添加类别信息到 COCO 数据集
    for root, dirs, files in os.walk(os.path.join(input_dir, 'labels')):
        for file in files:
            if file.endswith('.xml'):
                xml_path = os.path.join(root, file)
                tree = ET.parse(xml_path)
                root_elem = tree.getroot()

                for obj in root_elem.findall('.//object'):
                    category_name = obj.find('name').text
                    if category_name not in categories and category_name in useful_labels:
                        categories[category_name] = {"id": category_id, "name": category_name, "supercategory": "none"}
                        category_id += 1

    coco_data_train["categories"] = list(categories.values())
    coco_data_val["categories"] = list(categories.values())

    # 添加图像和标注信息
    image_id = 1
    annotation_id = 1
    added_images = set()

    for root, dirs, files in os.walk(os.path.join(input_dir, 'labels')):
        random.shuffle(files)  # 随机打乱文件顺序
        split_index = int(len(files) * split_ratio)

        for i, file in enumerate(files):
            if file.endswith('.xml'):
                xml_path = os.path.join(root, file)
                tree = ET.parse(xml_path)
                root_elem = tree.getroot()

                image_info = {
                    "id": image_id,
                    "width": int(root_elem.find('.//size/width').text),
                    "height": int(root_elem.find('.//size/height').text),
                    "file_name": file.replace('.xml', '.jpg'),
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": None
                }

                # 如果图像已经添加过，跳过
                if image_info["id"] in added_images:
                    continue

                annotations = []
                for obj in root_elem.findall('.//object'):
                    category_name = obj.find('name').text
                    if category_name in useful_labels:
                        category_id = categories[category_name]["id"]

                        # 计算调整比例，假设 VOC 数据集坐标是绝对坐标，需要将其调整到相对坐标
                        width = image_info["width"]
                        height = image_info["height"]

                        xmin = float(obj.find('.//bndbox/xmin').text) / width
                        ymin = float(obj.find('.//bndbox/ymin').text) / height
                        xmax = float(obj.find('.//bndbox/xmax').text) / width
                        ymax = float(obj.find('.//bndbox/ymax').text) / height

                        # 计算面积
                        area = (xmax - xmin) * (ymax - ymin)

                        # 计算四个坐标点，以左上角坐标为起始，顺时针依次选取的另外三个坐标点
                        x1, y1, w, h = xmin, ymin, (xmax - xmin), (ymax - ymin)
                        x2, y2 = x1 + w, y1
                        x3, y3 = x1 + w, y1 + h
                        x4, y4 = x1, y1 + h

                        annotation_info = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "segmentation": [x1, y1, x2, y2, x3, y3, x4, y4],
                            "area": area,
                            "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                            "iscrowd": 0  # Set to 0 for object instance annotations
                        }

                        annotations.append(annotation_info)
                        annotation_id += 1

                if i < split_index:
                    coco_data_train["images"].append(image_info)
                    coco_data_train["annotations"].extend(annotations)
                else:
                    coco_data_val["images"].append(image_info)
                    coco_data_val["annotations"].extend(annotations)

                added_images.add(image_info["id"])
                image_id += 1

    # 在 output_dir 下创建 train、val 和 annotations 文件夹
    output_train_dir = os.path.join(output_dir, 'train')
    output_val_dir = os.path.join(output_dir, 'val')
    output_annotations_dir = os.path.join(output_dir, 'annotations')

    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    os.makedirs(output_annotations_dir, exist_ok=True)

    # 保存训练集图像到 train 文件夹
    for image_info in coco_data_train["images"]:
        shutil.copy(os.path.join(input_dir, 'images', image_info["file_name"]), os.path.join(output_train_dir, image_info["file_name"]))

    # 保存验证集图像到 val 文件夹
    for image_info in coco_data_val["images"]:
        shutil.copy(os.path.join(input_dir, 'images', image_info["file_name"]), os.path.join(output_val_dir, image_info["file_name"]))

    # 将 COCO 数据保存为 JSON 文件
    with open(os.path.join(output_annotations_dir, 'csgcpdata_coco_train.json'), 'w') as json_file_train:
        json.dump(coco_data_train, json_file_train)

    with open(os.path.join(output_annotations_dir, 'csgcpdata_coco_val.json'), 'w') as json_file_val:
        json.dump(coco_data_val, json_file_val)

# 指定原始 Pascal VOC 数据集目录和有用标签
input_voc_directory = r'D:\DataSet\csgcpdata'
useful_labels = ['03030001']  # 替换成你的有用标签列表

# 指定输出的 output_dir 文件夹路径
output_dir = r'D:\PythonProject\Paddle\PaddleDetection\dataset\unspecified_coco'

# 调用函数进行转换
voc_to_coco_with_split(input_voc_directory, output_dir, useful_labels, split_ratio=0.8)

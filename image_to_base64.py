from datasets import load_dataset, load_from_disk
import base64
import io
import os
from PIL import Image

def load_and_encode_image(example):
    """将图片路径转换为Base64编码"""
    try:
        # 打开图片并转换为Base64
        with open(os.path.join(image_dir, example['image']), 'rb') as image_file:
            # 读取图片二进制数据
            image_data = image_file.read()
            # 转换为Base64编码
            base64_encoded = base64.b64encode(image_data).decode('utf-8')
            example['image'] = base64_encoded
        return example
    except Exception as e:
        print(f"处理图片 {example['image']} 时出错: {e}")
        # 出错时返回原示例（可根据需求修改）
        return example


def decode_and_get_info(base64_str):
    """将Base64编码转回图片，并返回图片信息"""
    # 解码Base64为二进制数据
    image_bytes = base64.b64decode(base64_str)
    
    # 计算原始图片的字节大小
    image_size_bytes = len(image_bytes)
    
    # 用PIL打开图片
    image = Image.open(io.BytesIO(image_bytes))
    
    # 获取图片尺寸（宽×高）
    width, height = image.size
    
    return image, image_size_bytes, width, height


# 1. 加载JSON数据集（假设JSON结构中包含"image_path"字段）
dataset = load_dataset('json', data_files='/mnt/haochen/datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json')['train']
image_dir = "/mnt/haochen/datasets/LLaVA-Pretrain"
save_path = "/mnt/haochen/datasets/encoded_llava_pretrain"

# 2. 处理数据集：将图片路径转换为Base64
# 对于大量图片，建议使用num_proc参数并行处理
encoded_dataset = dataset.map(
    load_and_encode_image,
    num_proc=16,
)

# 3. 保存处理后的数据集（会自动分片存储）
encoded_dataset.save_to_disk(save_path)

# 验证结果（可选）
print("处理完成的第一个样本Base64长度:", len(encoded_dataset[0]['image']))

# 加载处理后的数据集
dataset = load_from_disk(save_path)

# 打印前5张图片的Base64编码长度（字节数）
for i in range(5):
    image, size_bytes, width, height = decode_and_get_info(dataset[i]['image'])
    print(f"第{i+1}张图, {image.size} Base64长度: {len(dataset[i]['image'])} bytes")
import os
import clip
import torch
import numpy as np
from PIL import Image

# 初始化 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device=device)

def forward_modality(model, preprocess, data, flag):
    """
    根据数据类型（图像或文本）提取特征
    :param model: CLIP 模型
    :param preprocess: 图像预处理函数
    :param data: 输入数据（图像或文本）
    :param flag: 数据类型，'img' 或 'txt'
    :return: 提取的特征
    """
    device = next(model.parameters()).device
    if flag == 'img':
        data = preprocess(data).unsqueeze(0)
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        data = clip.tokenize(data)
        features = model.encode_text(data.to(device))
    else:
        raise TypeError("Invalid flag. Use 'img' or 'txt'.")
    return features

@torch.no_grad()
def calculate_clip_score(model, preprocess, first_data, second_data, first_flag='txt', second_flag='img'):
    """
    计算 CLIP 分数
    :param model: CLIP 模型
    :param preprocess: 图像预处理函数
    :param first_data: 第一种数据（文本或图像）
    :param second_data: 第二种数据（文本或图像）
    :param first_flag: 第一种数据的类型，'txt' 或 'img'
    :param second_flag: 第二种数据的类型，'txt' 或 'img'
    :return: CLIP 分数
    """
    first_features = forward_modality(model, preprocess, first_data, first_flag)
    second_features = forward_modality(model, preprocess, second_data, second_flag)

    # 归一化特征
    first_features = first_features / first_features.norm(dim=1, keepdim=True).to(torch.float32)
    second_features = second_features / second_features.norm(dim=1, keepdim=True).to(torch.float32)

    # 计算相似度分数
    score = (second_features * first_features).sum()
    return score

def compute_image_text_alignment(image_path, text):
    """
    计算图像和文本的 CLIP 分数
    :param image_path: 图像路径
    :param text: 文本
    :return: CLIP 分数
    """
    image = Image.open(image_path)
    score = calculate_clip_score(model, preprocess, first_data=text, second_data=image, first_flag='txt', second_flag='img').cpu().numpy()
    return score

def extract_prompt_index(filename):
    """
    从文件名中提取 prompt 索引
    :param filename: 文件名
    :return: prompt 索引
    """
    # 假设文件名格式为 scaleX_promptY_seedZ.png
    parts = filename.split('_')
    prompt_part = parts[1]  # 获取 promptY 部分
    prompt_index = int(prompt_part.replace('prompt', ''))  # 提取 Y
    return prompt_index

def compute_average_clip_score(image_folder, positive_personal_prompts):
    """
    计算文件夹内所有图像的平均 CLIP 分数
    :param image_folder: 图像文件夹路径
    :param positive_personal_prompts: 文本提示列表
    :return: 平均 CLIP 分数
    """
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    scores = []
    for image_file in image_files:
        try:
            prompt_index = extract_prompt_index(image_file)
            text = positive_personal_prompts[prompt_index]
            image_path = os.path.join(image_folder, image_file)
            score = compute_image_text_alignment(image_path, text)
            scores.append(score)
            print(f"Image: {image_file}, Prompt Index: {prompt_index}, CLIP Score: {score}")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    if scores:
        average_score = np.mean(scores)
        print(f"Average CLIP Score for {image_folder}: {average_score}")
        return average_score
    else:
        print(f"No valid images found in {image_folder}.")
        return None

def compute_clip_for_all_folders(base_folder, positive_personal_prompts):
    """
    计算 base_folder 下所有 Guidance_Scale 文件夹的平均 CLIP 分数
    :param base_folder: 包含 Guidance_Scale 文件夹的根目录
    :param positive_personal_prompts: 文本提示列表
    :return: 包含每个文件夹名称和平均 CLIP 分数的字典
    """
    results = {}  # 用于存储每个文件夹的结果

    # 遍历 Guidance_Scale=1 到 Guidance_Scale=7 的文件夹
    for i in range(1, 8):
        folder_name = f"Guidance_Scale={i}"
        folder_path = os.path.join(base_folder, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"文件夹 {folder_name} 不存在，跳过")
            results[folder_name] = None  # 如果文件夹不存在，存储为 None
            continue
        
        print(f"正在计算文件夹: {folder_name}")
        try:
            average_clip_score = compute_average_clip_score(folder_path, positive_personal_prompts)
            results[folder_name] = average_clip_score  # 存储结果
        except Exception as e:
            print(f"计算文件夹 {folder_name} 时出错: {e}")
            results[folder_name] = None  # 如果出错，存储为 None

    return results

# 示例使用
base_folder = "./"  # 替换为你的根目录路径
positive_personal_prompts = [
    'A fluffy white cat with blue eyes sitting gracefully on a windowsill, bathed in golden sunlight, with a serene garden visible through the window.',
    'A majestic orange cat with a crown, sitting on a throne in a medieval castle, surrounded by intricate tapestries and candlelight.',
    'A gray cat with green eyes, sitting on a wooden porch, with soft sunlight highlighting its fur and a blurred garden in the background.',
    'A white cat with round, expressive eyes, sitting on a leather armchair in a cozy library filled with books and warm lighting.',
    'A black cat sitting on a stack of old books in a dusty attic, its multicolored fur contrasting with the vintage surroundings, with beams of sunlight filtering through a small window.',
    'A fluffy gray-and-white cat with golden eyes sits curled up on a cozy knitted blanket by a crackling fireplace.',
    'A brown cat with soft fur and green eyes sits calmly on a rustic wooden table in a sunlit kitchen.',
    'A black-and-white cat with curious eyes sits on a wooden porch, surrounded by autumn leaves and soft sunlight.',
    'A sleek brown cat with golden eyes sits on a polished grand piano, its fur reflecting the soft light of the room.',
    'A sleek black cat with yellow eyes sits on a cobblestone street at dusk, its fur glowing under the light of a streetlamp.',
    'A majestic dog with striking blue eyes and a muscular build stands alert on a rocky cliff edge, its thick, wavy fur glowing in the golden hour sunlight.',
    'A graceful dog with silky, well-groomed fur and deep, soulful eyes sits calmly in a sunlit meadow, its alert ears perked up and a subtle smile hinting at its friendly nature.',
    'A lively dog with a lean, athletic physique dashes through a field of tall grass, its wagging tail and bright, inquisitive eyes capturing pure joy as sunlight streams through the blades.',
    'A contented dog with soft, fluffy fur and gentle, half-closed eyes lies comfortably on a cozy couch.',
    'A curious dog with a finely detailed coat marked by subtle brindle patterns carefully sniffs the ground in an autumn forest.',
    'A dog with a sleek, black coat and bright, alert eyes runs through a shallow stream, water splashing around its paws, with sunlight reflecting off the ripples.',
    'A dog with a short, brindle coat and a strong jawline sits by a campfire, its eyes reflecting the flickering flames and ears twitching at the sound of crackling wood.',
    'A dog with a thick, double-layered coat stands in a snowy field, its breath visible in the cold air and snowflakes clinging to its fur, looking intently at something ahead.',
    'A dog with a curly, white coat and a pink nose plays in a field of wildflowers, its tongue out and tail wagging energetically, surrounded by vibrant colors.',
    'A lively dog with a glossy, golden coat and a slightly tilted head looks up with curious eyes, its ears perked and nose twitching, standing in a sunlit garden filled with vibrant flowers.',
]

# 计算所有文件夹的平均 CLIP 分数
results = compute_clip_for_all_folders(base_folder, positive_personal_prompts)

# 统一打印所有文件夹的平均 CLIP 分数
print("\n所有文件夹的平均 CLIP 分数：")
for folder_name, score in results.items():
    if score is not None:
        print(f"{folder_name}: {score}")
    else:
        print(f"{folder_name}: 无法计算")
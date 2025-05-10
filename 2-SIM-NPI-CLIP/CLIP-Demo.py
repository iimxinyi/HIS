import clip
import torch
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device=device)

def forward_modality(model, preprocess, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        data = preprocess(data).unsqueeze(0)
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        data = clip.tokenize(data)
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    # print(flag, features.shape)
    return features

@torch.no_grad()
def calculate_clip_score(model, preprocess, first_data, second_data, first_flag='txt', second_flag='img'):
    first_features = forward_modality(model, preprocess, first_data, first_flag)
    second_features = forward_modality(model, preprocess, second_data,second_flag)

    # normalize features
    first_features = first_features / first_features.norm(dim=1, keepdim=True).to(torch.float32)
    second_features = second_features / second_features.norm(dim=1, keepdim=True).to(torch.float32)

    # calculate scores
    # score = logit_scale * (second_features * first_features).sum()
    score = (second_features * first_features).sum()
    return score

def compute_image_text_alignment(image_path, text):
    global model, preprocess

    image = Image.open(image_path)
    score = calculate_clip_score(model, preprocess, first_data=text, second_data=image, first_flag='txt', second_flag='img').cpu().numpy()
    return score


# Demo
image_path = "./Public0_Personal2_CommonStep1.png"
text = "A fluffy white cat with blue eyes sitting gracefully on a windowsill, bathed in golden sunlight, with a serene garden visible through the window."
clip_score = compute_image_text_alignment(image_path, text)
print(clip_score)


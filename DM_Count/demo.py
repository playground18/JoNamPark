import torch
from models import vgg19
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import os
from tqdm import tqdm 

model_path = "C:/jnp/DM_Count/pretrained_models/model_qnrf.pth"

device = torch.device('cpu')
# device = torch.device('cuda:0')

model = vgg19()
model.to(device)

model.load_state_dict(torch.load(model_path, device))
#model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()


def predict(inp, image_name):
    inp = cv2.imread(inp)
    inp = Image.fromarray(inp.astype('uint8'), 'RGB')   
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    inp = inp.to(device)
    with torch.set_grad_enabled(False):
        outputs, _ = model(inp)
    count = torch.sum(outputs).item()
    vis_img = outputs[0, 0].cpu().numpy()
    # normalize density map values from 0 to 1, then map it to 0-255.
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_HSV)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    # increase the resolution of the density map
    vis_img = cv2.resize(vis_img, None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)

    # Add count as a caption in the upper right corner of the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text = f"Count: {round(count)}"
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    height, width, _ = vis_img.shape
    margin = 10
    x = width - text_size[0] - margin
    y = text_size[1] + margin
    cv2.putText(vis_img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    result_folder = "./media/results"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    result_path = os.path.join(result_folder, f"{os.path.splitext(os.path.basename(image_name))[0]}_density_map.png")
    cv2.imwrite(result_path, vis_img)

"""     result_folder = "results"
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    result_path = os.path.join(result_folder, f"{os.path.splitext(os.path.basename(image_name))[0]}_density_map.png")
    cv2.imwrite(result_path, vis_img) """



input_folder = "C:/jnp/data/0-20"
images = os.listdir(input_folder)
images = [img for img in images if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png")]
images.sort()

for i, image in tqdm(enumerate(images), total=len(images), desc='Processing images'):
    path = os.path.join(input_folder, image)
    predict(path, image)
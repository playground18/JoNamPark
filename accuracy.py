from ultralytics import YOLO
import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import os
from DM_Count.models import vgg19
import pandas as pd
from accuracy_plotting import count_plot, accuracy_plot


#아래 설정들을 바꿔서 테스트할 수 있습니다.
root_path = 'C:/jnp/'
model_path = 'C:/jnp/DM_Count/pretrained_models/model_qnrf.pth'
detection_threshold = 0
IMAGE_FILE_PRINTING = True
device = torch.device('cpu')

#### Model Setting ####

model_yolo = YOLO('yolov8x.pt')
model_dmcount = vgg19()
model_dmcount.to(device)

model_dmcount.load_state_dict(torch.load(model_path, device))

model_dmcount.eval()
########################

data_path = os.path.join(root_path+'data')
images_folders = [f.name for f in os.scandir(data_path) if f.is_dir()]
ground_truth = []
cnt_yolo = []
cnt_dm_count = []
csv_paths = []

result_folder = os.path.join(root_path+'result/')
if not os.path.exists(result_folder):
	os.makedirs(result_folder)

#vis_img에 count:  텍스트를 작성
def write_count_on_image(vis_img, count):
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

	return vis_img


def get_ground_truth(excel_file_path):
	# 엑셀 파일 읽기
	df = pd.read_excel(excel_file_path)

	# 'count' 열의 데이터를 리스트로 추출
	return df['count'].tolist()


def predict_yolo(inp, folder, image, output_path):
	print(f'yolo {folder}/{image}')

	results = model_yolo(inp)

	#사람 이라고 판단한 객체의 bounding box 위치 좌표 저장용
	predicted_boxes = []
	
	for result in results:
		for r in result.boxes.data.tolist():
			x1, y1, x2, y2, score, class_id = r
			if class_id == 0 and score >= detection_threshold:
				predicted_boxes.append([x1, y1, x2, y2])

	cnt = len(predicted_boxes)

	# {root_path}/result/{folder_name}/yolo_{image}에 bbox 포함되고, count labeling된 이미지 저장
	if IMAGE_FILE_PRINTING:
		#이미지에 bounding box 표시
		for person in predicted_boxes:
			x1, y1, x2, y2 = map(int, map(float, person[:4]))
			inp = cv2.rectangle(inp, (x1, y1), (x2, y2), (255, 0, 0), 2)
		#count labeling
		inp = write_count_on_image(inp, cnt)
	

		result_image_path = os.path.join(output_path, 'yolo_'+image)
		cv2.imwrite(result_image_path, inp)

	cnt_yolo.append(cnt)


def predict_dmcount(inp, folder, image, output_path):
	print(f'dm_count {folder}/{image}')
	inp = Image.fromarray(inp.astype('uint8'), 'RGB')   
	inp = transforms.ToTensor()(inp).unsqueeze(0)
	inp = inp.to(device)
	with torch.set_grad_enabled(False):
		outputs, _ = model_dmcount(inp)
	cnt = torch.sum(outputs).item()

	if IMAGE_FILE_PRINTING:
		vis_img = outputs[0, 0].cpu().numpy()
		# normalize density map values from 0 to 1, then map it to 0-255.
		vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
		vis_img = (vis_img * 255).astype(np.uint8)
		vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_HSV)
		vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

		# increase the resolution of the density map
		vis_img = cv2.resize(vis_img, None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)

		write_count_on_image(vis_img, cnt)
			
		result_image_path = os.path.join(output_path, 'dm_count_'+image)

		cv2.imwrite(result_image_path, vis_img)

	cnt_dm_count.append(round(cnt))

for images_folder in images_folders:
	ground_truth = []
	cnt_yolo = []
	cnt_dm_count = []

	folder_path = os.path.join(data_path, images_folder)
	all_files = os.listdir(folder_path)
	image_files = [img for img in all_files if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png")]

	sorted_image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))

	excel_file_path = os.path.join(folder_path, images_folder+'.xlsx')

	ground_truth = get_ground_truth(excel_file_path)

	output_path = os.path.join(result_folder, images_folder)
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	for image in sorted_image_files:
		path = os.path.join(folder_path, image)
		inp = cv2.imread(path)
		predict_dmcount(inp, images_folder, image, output_path)

	for image in sorted_image_files:
		path = os.path.join(folder_path, image)
		inp = cv2.imread(path)
		predict_yolo(inp, images_folder, image, output_path)

	csv_path = os.path.join(output_path, f'{images_folder}.csv')
	csv_paths.append(csv_path)

	with open(csv_path, 'w') as f:
		f.write('order,ground_truth,yolo,dm_count\n')
		for i in range(len(sorted_image_files)):
			f.write(f"{i+1},{ground_truth[i]},{cnt_yolo[i]},{cnt_dm_count[i]}\n")

#통계적으로 보기 좋게 csv파일들 경로저장
csv_paths_list_path = os.path.join(result_folder, 'csv_paths.csv')
with open(csv_paths_list_path, 'w') as f:
	f.write('folder,csv_path\n')
	for i in range(len(csv_paths)):
		f.write(f'{images_folders[i]},{csv_paths[i]}\n')

count_plot(csv_paths, result_folder)
accuracy_plot(csv_paths, result_folder)

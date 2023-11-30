## Title: Counting test with yolov8 and DMCount
## 개요
Pretrain된 YOLOv8x, DM_Count(qnrf) 활용하여 data folder안의 폴더별 이미지에서
사람 객체 수를 모델별로 count하여 실제 count(ground truth)와 비교하여 봅니다.

예시로 주어진 데이터는 실제 사람 수가 0-20/20-40/40-60/60-80/80-100인 test set을 이용하였습니다.

## using
accuracy.py

root_path = 'C:/jnp/'
model_path = 'C:/jnp/DM_Count/pretrained_models/model_qnrf.pth'
detection_threshold = 0
IMAGE_FILE_PRINTING = True
device = torch.device('cpu')

경로/옵션을 변경하여서 사용

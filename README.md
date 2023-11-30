#### Counting test with yolov8 and DMCount

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

## directory structure example
```
ROOT (root_path)
	├ DM_Count	
	│	├ models.py (using vgg19)
	│	└ pretrained_models/{dm_count model} (model_path)
	│    
	├ data
	│	├ {folder1} (카테고리별 image 데이터)
	│	│	├ {image1.png} (.jpg/.jpeg도 가능)
	│	│	├ {image2.png}
	│	│	├ {image3.png}
	│	│		...
	│	│	└ {folder1.xlsx}
	│	│
	│	├ {folder2}
	│	│	├ {image1.png}
	│	│	├ {image2.png}
	│	│	├ {image3.png}
	│	│		...
	│	│	└ {folder1.csv}
	│	│
	│	...
	│
	├ result (result_folder, 없으면 생성됨)
	│	├ {folder1} (bbox, counting labeling된 image 저장용, IMAGE_FILE_PRINTING의 값을 변경해서 결과출력/미출력)
	│	│	├ {yolo_image1.png} ()
	│	│	├ {dm_count_image1.png}
	│	│		...
	│	│	└ {folder1.csv}
	│	├ {folder2}
	│	│	├ {yolo_image1.png} 
	│	│	├ {dm_count_image1.png}
	│	│		...
	│	│	└ {folder2.csv}
	│	...
	│	├ {csv_paths.csv} (폴더명,csv위치)(accuracy_plotting에서 이 위치를 이용하여 plot을 생성함)
	│	├ {accuracy.png} (폴더별 yolo,dm_count의 accuracy 평균)
	│	└ {countDiff.png} (폴더별 이미지들의 ground_truth/yolov8/dm_count의 counting) 
	│	
	├ accuracy.py (main code)
	└ accuracy_plotting.py (plot생성에 필요한 코드)
```

## 사용한 코드

이 프로젝트에서는 아래 사용자의 코드를 참고하였습니다.

- [ultralytics](https://github.com/ultralytics/ultralytics)
- [DM-Count][https://github.com/cvlab-stonybrook/DM-Count]
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#폴더별로 
def count_plot(csv_paths, result_folder):

    csv_paths_df = pd.read_csv(csv_paths)

    yolo = {}
    dm_count = {}
    ground_truth = {}

    for index, row in csv_paths_df.iterrows():
        folder_name = row['folder']
        csv_path = row['csv_path']

        # CSV 파일 읽기
        df = pd.read_csv(csv_path)

        yolo[folder_name]=df['yolo'].tolist()
        dm_count[folder_name]=df['dm_count'].tolist()
        ground_truth[folder_name]=df['ground_truth'].tolist()

    # 데이터의 개수
    num_folders = len(yolo)

    # 5개마다 가로로 나열하도록 열의 개수 설정
    num_cols = 5

    # 행, 열 개수 계산
    num_rows = (num_folders + num_cols - 1) // num_cols

    # 서브플롯 설정
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows), sharex=True)

    # 각 폴더에 대한 데이터를 그래프로 표시
    for i, (folder, data) in enumerate(yolo.items()):
        if num_rows == 1:
            ax = axes[i]
        else:
            row, col = divmod(i, num_cols)
            ax = axes[row, col]
        data2 = dm_count[folder]
        gt = ground_truth[folder]
        marker_size = 35 / len(data)
        ax.plot(range(len(data)), data, color='orange',linewidth=0.5,marker = 'o', markersize = marker_size)
        ax.plot(range(len(data2)), data2, color='green',linewidth=0.5,marker = 'o', markersize = marker_size)
        ax.plot(range(len(gt)), gt, color='black',linewidth=0.5,marker = 'o', markersize = marker_size)
        ax.set_ylabel('Count')
        ax.set_xlabel('images')
        ax.set_title(folder)
        ax.set_xticks(range(0,len(data),len(data)//10))
        ax.set_xticklabels(range(1,len(data)+1, (len(data)+1)//10),rotation=45)



    # 빈 서브플롯 숨기기
    for i in range(num_folders, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    plt.suptitle('Count Diff per dataset\nOrange(YOLOv8), Green(DM_Count), Black(GT)')
    plt.tight_layout()
    plt.savefig(result_folder+'countDiff.png')

def accuracy_plot(csv_paths, result_folder):
    csv_paths_df = pd.read_csv(csv_paths)

    # 전체 데이터를 저장할 리스트 초기화
    all_accuracies_yolo = []
    all_accuracies_dm_count = []
    all_folder_names = []

    for index, row in csv_paths_df.iterrows():
        folder_name = row['folder']
        csv_path = row['csv_path']

        # CSV 파일 읽기
        df = pd.read_csv(csv_path)

        # 각 모델별 정확도 계산
        yolo_accuracy = (1-(abs(df['ground_truth'] - df['yolo'])/df['ground_truth'])).mean()*100
        dm_count_accuracy = (1-(abs(df['ground_truth'] - df['dm_count'])/df['ground_truth'])).mean()*100


        # 결과 저장
        all_folder_names.append(folder_name)
        all_accuracies_yolo.append(yolo_accuracy)
        all_accuracies_dm_count.append(dm_count_accuracy)

    # 그래프 그리기
    print(all_accuracies_dm_count, all_accuracies_yolo)
    plt.figure(figsize=(10, 5))

    plt.plot(all_folder_names, all_accuracies_yolo, marker='o', label='YOLOv8', color='orange')
    plt.plot(all_folder_names, all_accuracies_dm_count, marker='o', label='DM Count', color='green')

    plt.title('Model Accuracy')
    plt.xlabel('Folder Name')
    plt.ylabel('Accuracy')
    plt.ylim(0, 100.0)  # y축을 백분율로 표시하기 위해 y축 제한 설정
    plt.legend()


    plt.savefig(result_folder+'accuracy.png')

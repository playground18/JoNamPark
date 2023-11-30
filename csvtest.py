import os
from accuracy_plotting import count_plot, accuracy_plot

root_path = 'C:/jnp/'
# data_path = os.path.join(root_path+'data')
# images_folders = [f.name for f in os.scandir(data_path) if f.is_dir()]
result_folder = os.path.join(root_path+'result/')

# csv_paths = ['C:/jnp/result/0-20/0-20.csv',
#              'C:/jnp/result/20-40/20-40.csv',
#              'C:/jnp/result/40-60/40-60.csv',
#              'C:/jnp/result/60-80/60-80.csv',
#              'C:/jnp/result/80-100/80-100.csv'
#             ]

# #통계적으로 보기 좋게 csv파일들 경로저장
# csv_paths_list_path = os.path.join(result_folder, 'csv_paths.csv')
# with open(csv_paths_list_path, 'w') as f:
# 	f.write('folder,csv_path\n')
# 	for i in range(len(csv_paths)):
# 		f.write(f'{images_folders[i]},{csv_paths[i]}\n')
count_plot('C:/jnp/result/csv_paths.csv', result_folder)
accuracy_plot('C:/jnp/result/csv_paths.csv', result_folder)
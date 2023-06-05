import os
import shutil
import scipy.io

data_source_root = r'C:\dpm\CrowdCounting-P2PNet-main\ShanghaiTech\part_B\test_data'
output_dir = r'C:\dpm\CrowdCounting-P2PNet-main\DATA_ROOT'
dataset_part = 'test'
output_dir_data_lists = output_dir
output_dir = os.path.join(output_dir, dataset_part)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

source_images_dir = os.path.join(data_source_root, 'images')
file_names = os.listdir(source_images_dir)
data_list = []
for file_name in file_names:
    shutil.copy(os.path.join(source_images_dir, file_name), output_dir)

source_labels_dir = os.path.join(data_source_root, 'ground-truth')
file_names = os.listdir(source_labels_dir)
for file_name in file_names:
    new_file_name = file_name[3:].split('.mat')[0] + ".txt"  # GT_IMG_12.mat -> IMG_12.mat -> IMG_12 -> IMG_12.txt
    new_file_name = os.path.join(output_dir, new_file_name)
    with open(new_file_name, 'a') as txt_file:
        mat_file = scipy.io.loadmat(os.path.join(source_labels_dir, file_name))
        for pair in mat_file['image_info'][0][0][0][0][0]:
            txt_file.write(f'{pair[0]} {pair[1]}\n')

    img_name = new_file_name[:-3] + 'jpg'
    data_list.append((img_name, new_file_name))


data_list_filename = os.path.join(output_dir_data_lists, f"{dataset_part}.list")
with open(data_list_filename, 'a') as file:
    for pair in data_list:
        file.write(f'{pair[0]} {pair[1]}\n')

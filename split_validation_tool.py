import os
import random
import shutil

def move_random_files(src_folder, dest_folder, percentage=0.1):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    num_files_to_move = int(len(all_files) * percentage)
    files_to_move = random.sample(all_files, num_files_to_move)
    for file_name in files_to_move:
        src_file = os.path.join(src_folder, file_name)
        dest_file = os.path.join(dest_folder, file_name)
        shutil.move(src_file, dest_file)

src_folder = 'raw'
dest_folder = os.path.join(src_folder, 'val')
move_random_files(src_folder, dest_folder)


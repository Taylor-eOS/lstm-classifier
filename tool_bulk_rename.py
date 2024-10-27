import os

def rename_files():
    folder_path = input("Enter the folder path: ")
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    counter = 1
    for filename in sorted(files):
        base, ext = os.path.splitext(filename)
        if base.endswith('_a'):
            suffix = '_a'
        elif base.endswith('_b'):
            suffix = '_b'
        else:
            continue
        new_name = f"{counter}{suffix}{ext}"
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))
        counter += 1
rename_files()


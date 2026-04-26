import os
import shutil
import glob

base_dir = r"E:\AI-TOD"
train_img = os.path.join(base_dir, "train", "images")
val_img = os.path.join(base_dir, "val", "images")
trainval_img = os.path.join(base_dir, "trainval", "images")

print("Initializing trainval directory in", trainval_img)
os.makedirs(trainval_img, exist_ok=True)

def link_files(src_dir):
    if not os.path.exists(src_dir):
        print(f"Directory {src_dir} does not exist. Skipping.")
        return
    
    files = glob.glob(os.path.join(src_dir, "*.*"))
    count = 0
    for img_path in files:
        dst_path = os.path.join(trainval_img, os.path.basename(img_path))
        if not os.path.exists(dst_path):
            try:
                os.link(img_path, dst_path)
            except Exception:
                shutil.copy2(img_path, dst_path)
        count += 1
    print(f"Processed {count} images from {src_dir}")

link_files(train_img)
link_files(val_img)
print("Finished setting up AI-TOD trainval images!")

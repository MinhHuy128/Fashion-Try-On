import os
import shutil

def flatten_and_merge(src, dest):
    """Recursively moves folders like 'test_color', 'train_img' to dest"""
    for root, dirs, files in os.walk(src, topdown=False):
        dir_name = os.path.basename(root)
        # If this directory is a target ACGPN folder type (e.g., train_color, test_img)
        if dir_name in ['train_color', 'train_img', 'train_label', 'train_pose', 'train_colormask', 'train_edge',
                        'test_color', 'test_img', 'test_label', 'test_pose', 'test_colormask', 'test_edge']:
            target_dir = os.path.join(dest, dir_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # Move all files inside this specific directory to the target directory
            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(target_dir, file)
                if not os.path.exists(dest_file):
                    shutil.move(src_file, dest_file)
                    
            # After moving files, the directory should be empty or only contain subdirs
            
def consolidate_data():
    base_dir = "data"
    target_dir = os.path.join(base_dir, "ACGPN_raw")
    os.makedirs(target_dir, exist_ok=True)
    
    source_dirs = [
        os.path.join(base_dir, "archive"),
        os.path.join(base_dir, "Data_preprocessing")
    ]
    
    print("Starting smart data consolidation...")
    for src in source_dirs:
        if os.path.exists(src):
            flatten_and_merge(src, target_dir)
            
            # Clean up the old directory safely
            try:
                shutil.rmtree(src)
                print(f"Removed temporary directory: {src}")
            except Exception as e:
                print(f"Could not remove {src}: {e}")
                
    print("Data consolidation complete! All folders are properly merged in data/ACGPN_raw")

if __name__ == "__main__":
    consolidate_data()

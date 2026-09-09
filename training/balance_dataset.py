import os
import shutil
import random
from collections import defaultdict

SOURCE_DIR = "../data/processed_v7/train/fake"
TARGET_DIR = "../data/processed_v7_balanced/train/fake"
TARGET_COUNT_PER_SOURCE = 2000

def get_source_from_filename(filename):
    fname = filename.lower()
    if 'gemini' in fname:
        return 'gemini'
    elif 'midjourney' in fname or 'mj_' in fname or 'genimgmj' in fname:
        return 'midjourney'
    elif 'cifake' in fname:
        return 'cifake'
    elif 'diffdb' in fname or 'diffusion' in fname:
        return 'stable_diffusion'
    elif 'genimgsd' in fname:
        return 'stable_diffusion'
    return 'unknown'

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    source_files = defaultdict(list)
    
    for fname in os.listdir(SOURCE_DIR):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            continue
        source = get_source_from_filename(fname)
        if source != 'unknown':
            source_files[source].append(fname)
    
    print("Source distribution:")
    for source, files in source_files.items():
        print(f"  {source}: {len(files)} images")
    
    for source, files in source_files.items():
        random.shuffle(files)
        
        if len(files) < TARGET_COUNT_PER_SOURCE:
            selected = files * (TARGET_COUNT_PER_SOURCE // len(files)) + files[:TARGET_COUNT_PER_SOURCE % len(files)]
            print(f"  {source}: Upsampling from {len(files)} to {TARGET_COUNT_PER_SOURCE} (x{len(selected)/len(files):.1f})")
        else:
            selected = files[:TARGET_COUNT_PER_SOURCE]
            print(f"  {source}: Downsampling from {len(files)} to {TARGET_COUNT_PER_SOURCE}")
        
        for i, fname in enumerate(selected):
            src = os.path.join(SOURCE_DIR, fname)
            base, ext = os.path.splitext(fname)
            dst_name = f"{source}_{i:05d}{ext}"
            dst = os.path.join(TARGET_DIR, dst_name)
            shutil.copy2(src, dst)
    
    print(f"\nBalanced dataset created at {TARGET_DIR}")
    print(f"Total images: {sum(len(files) for files in source_files.values()) * (TARGET_COUNT_PER_SOURCE // min(len(f) for f in source_files.values()))}")

if __name__ == "__main__":
    main()
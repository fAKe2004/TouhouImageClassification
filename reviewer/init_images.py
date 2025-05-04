import os

DATA_DIR = 'data'

def initialize_pending():
    print("开始初始化图像状态...")
    
    for label in os.listdir(DATA_DIR):
        label_dir = os.path.join(DATA_DIR, label)
        
        if not os.path.isdir(label_dir):
            continue
            
        print(f"\n处理分类目录: {label}")
        processed = 0
        
        for filename in os.listdir(label_dir):
            # 跳过参考图像
            if filename == "1.jpg":
                print(f"跳过参考图像: {filename}")
                continue
                
            src_path = os.path.join(label_dir, filename)
            
            # 仅处理普通文件
            if not os.path.isfile(src_path):
                continue
                
            # 分割文件名和扩展名
            basename, ext = os.path.splitext(filename)
            
            # 已经是待审查状态则跳过
            if ext == ".pending":
                print(f"保持待审查: {filename}")
                continue
                
            # 构建新文件名
            new_filename = f"{basename}.pending"
            dest_path = os.path.join(label_dir, new_filename)
            
            try:
                os.rename(src_path, dest_path)
                print(f"转换成功: {filename} -> {new_filename}")
                processed += 1
            except Exception as e:
                print(f"错误处理 {filename}: {str(e)}")
                
        print(f"完成处理 {label} 目录，转换 {processed} 个文件")
    
    print("\n初始化完成！")

if __name__ == "__main__":
    initialize_pending()

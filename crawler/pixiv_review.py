import os
import json
import time
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image

DATA_DIR = 'data'
REFERENCE_NAME = 'reference.jpg'
PROGRESS_FILE = 'review_progress.json'

class ImageReviewSystem:
    def __init__(self, master):
        self.master = master
        self.labels = []
        self.current_label = None
        self.images = []
        self.index = 0
        self.history = []
        self.progress = {'completed': []}

        self.init_ui()
        self.load_progress()
        self.prepare_labels()
        self.start_review()

    def init_ui(self):
        self.master.title("智能图片审查系统")
        self.master.geometry("1100x850")
        self.master.protocol("WM_DELETE_WINDOW", self.safe_exit)

        # 状态显示栏
        self.status_bar = tk.Frame(self.master, bg="#f0f0f0", height=40)
        self.status_bar.pack(fill=tk.X, pady=5)
        self.status_text = tk.Label(self.status_bar, bg="#f0f0f0", font=('微软雅黑', 12))
        self.status_text.pack(side=tk.LEFT, padx=20)

        # 图片显示区域
        self.img_frame = tk.Frame(self.master)
        self.img_frame.pack(pady=15)
        
        # 参考图面板
        self.ref_panel = tk.Label(self.img_frame)
        self.ref_panel.pack(side=tk.LEFT, padx=30)
        
        # 待审图面板
        self.review_panel = tk.Label(self.img_frame)
        self.review_panel.pack(side=tk.RIGHT, padx=30)

        # 操作按钮组
        self.control_frame = tk.Frame(self.master)
        self.control_frame.pack(pady=20)
        
        self.btn_rollback = tk.Button(
            self.control_frame, text="回退 (B)", 
            command=self.rollback_action, width=12, height=2
        )
        self.btn_valid = tk.Button(
            self.control_frame, text="有效 (空格)", 
            command=lambda: self.process_image(valid=True), width=12, height=2
        )
        self.btn_invalid = tk.Button(
            self.control_frame, text="无效 (N)", 
            command=lambda: self.process_image(valid=False), width=12, height=2
        )

        self.btn_rollback.pack(side=tk.LEFT, padx=15)
        self.btn_valid.pack(side=tk.LEFT, padx=15)
        self.btn_invalid.pack(side=tk.LEFT, padx=15)

        # 绑定快捷键
        self.master.bind('<space>', lambda e: self.process_image(valid=True))
        self.master.bind('<n>', lambda e: self.process_image(valid=False))
        self.master.bind('<N>', lambda e: self.process_image(valid=False))
        self.master.bind('<b>', lambda e: self.rollback_action())
        self.master.bind('<B>', lambda e: self.rollback_action())

    # === 核心功能模块 ===
    def load_progress(self):
        """加载进度数据"""
        try:
            if os.path.exists(PROGRESS_FILE):
                with open(PROGRESS_FILE, 'r') as f:
                    data = json.load(f)
                    self.progress = data.get('progress', {'completed': []})
                    self.labels = data.get('labels_queue', [])
                    self.history = data.get('history', [])
        except Exception as e:
            messagebox.showerror("加载错误", f"进度文件读取失败: {str(e)}")

    def save_progress(self):
        """保存当前进度"""
        progress_data = {
            'progress': {
                'completed': self.progress['completed'],
                'current_label': self.current_label,
                'current_index': self.index
            },
            'labels_queue': self.labels.copy(),
            'history': self.history[-50:]
        }
        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress_data, f)
        except Exception as e:
            messagebox.showerror("保存错误", f"进度保存失败: {str(e)}")

    def prepare_labels(self):
        """准备待处理标签队列"""
        all_labels = sorted(
            [d for d in os.listdir(DATA_DIR) 
            if os.path.isdir(os.path.join(DATA_DIR, d))],
            key=lambda x: x.lower()
        )
        self.labels = [
            lbl for lbl in all_labels 
            if lbl not in self.progress['completed']
        ]
        if self.current_label and self.current_label in self.labels:
            self.labels.remove(self.current_label)
            self.labels.insert(0, self.current_label)

    def load_images(self):
        """加载当前标签的图片列表"""
        if not self.current_label:
            return []
        
        label_dir = os.path.join(DATA_DIR, self.current_label)
        try:
            return sorted([
                f for f in os.listdir(label_dir)
                if f.endswith('.jpg') 
                and f != REFERENCE_NAME
                and not f.endswith('.invalid')
            ], key=lambda x: int(x.split('.')[0]))
        except:
            return []

    def update_display(self):
        """更新界面显示"""
        # 状态文本
        status = f"{self.current_label} | 进度: {self.index+1}/{len(self.images)}"
        if self.history:
            status += f" | 可回退: {len(self.history)}"
        self.status_text.config(text=status)

        # 显示参考图
        ref_path = os.path.join(DATA_DIR, self.current_label, REFERENCE_NAME)
        if os.path.exists(ref_path):
            ref_img = ImageTk.PhotoImage(Image.open(ref_path).resize((500,500)))
            self.ref_panel.config(image=ref_img)
            self.ref_panel.image = ref_img

        # 显示待审图
        if self.index < len(self.images):
            img_path = os.path.join(DATA_DIR, self.current_label, self.images[self.index])
            if os.path.exists(img_path):
                rev_img = ImageTk.PhotoImage(Image.open(img_path).resize((500,500)))
                self.review_panel.config(image=rev_img)
                self.review_panel.image = rev_img

    def process_image(self, valid=True):
        """处理图片标记操作"""
        if not self.images or self.index >= len(self.images):
            return

        current_file = self.images[self.index]
        action_type = 'valid' if valid else 'invalid'
        
        # 记录操作历史
        self.record_action({
            'action': action_type,
            'file': current_file,
            'prev_state': {
                'label': self.current_label,
                'index': self.index,
                'labels': self.labels.copy()
            }
        })

        # 执行文件操作
        if not valid:
            src = os.path.join(DATA_DIR, self.current_label, current_file)
            os.rename(src, f"{src}.invalid")

        # 移动到下一项
        self.index += 1
        self.handle_transition()

    def record_action(self, action_info):
        """记录操作历史"""
        full_record = {
            'timestamp': time.time(),
            **action_info,
            'images_list': self.images.copy()
        }
        self.history.append(full_record)

    def handle_transition(self):
        """处理状态转移"""
        if self.index < len(self.images):
            self.update_display()
            self.save_progress()
            return

        # 完成当前标签
        self.progress['completed'].append(self.current_label)
        self.save_progress()
        
        # 切换到下一个标签
        self.prepare_labels()
        if self.labels:
            self.current_label = self.labels.pop(0)
            self.images = self.load_images()
            self.index = 0
            self.update_display()
        else:
            self.complete_review()

    def rollback_action(self):
        """执行回退操作"""
        if not self.history:
            messagebox.showinfo("提示", "没有可回退的操作历史")
            return

        last_action = self.history.pop()
        
        # 恢复文件状态
        if last_action['action'] == 'invalid':
            src = os.path.join(
                DATA_DIR, 
                last_action['prev_state']['label'],
                last_action['file'] + '.invalid'
            )
            dst = src[:-8]  # 移除.invalid后缀
            if os.path.exists(src):
                os.rename(src, dst)

        # 恢复程序状态
        self.current_label = last_action['prev_state']['label']
        self.labels = last_action['prev_state']['labels']
        self.images = last_action['images_list']
        self.index = last_action['prev_state']['index']
        
        # 强制刷新界面
        self.prepare_labels()
        self.update_display()
        self.save_progress()

    def complete_review(self):
        """完成全部审查"""
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        messagebox.showinfo("完成", "所有图片审查已完成！")
        self.master.destroy()

    def safe_exit(self):
        """安全退出处理"""
        if messagebox.askokcancel("退出", "是否保存进度并退出？"):
            self.save_progress()
            self.master.destroy()

    # === 辅助方法 ===
    def start_review(self):
        """启动审查流程"""
        if not self.labels:
            self.complete_review()
            return

        self.current_label = self.labels.pop(0)
        self.images = self.load_images()
        self.index = 0
        self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageReviewSystem(root)
    root.mainloop()
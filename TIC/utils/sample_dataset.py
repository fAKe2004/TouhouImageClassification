import os
import shutil
import random
import argparse

from TIC.utils.parameter import *

def sample_dataset(src : str, dest : str, p : float):
    os.makedirs(dest, exist_ok=True)
    for label in os.listdir(src):
        if os.path.isdir(os.path.join(src, label)):
            try:
                os.removedirs(os.path.join(dest, label))
            except:
                pass
            os.makedirs(os.path.join(dest, label), exist_ok=True)
            file = os.listdir(os.path.join(src, label))
            n = len(file)
            m = int(n * p)
            file_chosen = random.sample(file, m)
            for f in file_chosen:
                shutil.copy(os.path.join(src, label, f), os.path.join(dest, label, f))

def add_reference(dest: str, ref : str, ref_num : str = "0.jpg"):
    for label in os.listdir(dest):
        if os.path.isdir(os.path.join(dest, label)):
            ref_file = None
            for f in os.listdir(os.path.join(ref, label)):
                if os.path.splitext(f)[1] != '.invalid':
                    ref_file = os.path.join(ref, label, f)
                    break
            shutil.copy(ref_file, os.path.join(dest, label, ref_num))

def del_reference(dest: str, ref_num : str = "0.jpg"):
    for label in os.listdir(dest):
        if os.path.isdir(os.path.join(dest, label)):
            try:
                os.remove(os.path.join(dest, label, ref_num))
            except:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--dest', type=str, required=True)
    parser.add_argument('--ref', type=str, default = TEST_DIR)
    parser.add_argument('-p', type=float, default = 0.01)
    parser.add_argument('-d', action='store_true', default = False)
    args = parser.parse_args()

    if args.src:
        sample_dataset(args.src, args.dest, args.p)

    if args.ref and not args.d:
        add_reference(args.dest, args.ref)

    if args.d:
        del_reference(args.dest)

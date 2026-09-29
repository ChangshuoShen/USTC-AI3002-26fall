#!/usr/bin/env python3
import numpy as np

def process_file_to_array(filename):
    # 打开文件并按行读取
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 替换每行中的 '#' 为 '6'
    processed_lines = [line.rstrip('\n').replace('#', '6') for line in lines]
    # 将列表转换为 numpy 数组
    arr = np.array(processed_lines)
    return arr

def main():
    filename = "data.txt"
    arr = process_file_to_array(filename)
    print(arr)

if __name__ == "__main__":
    main()

import os

import pandas as pd


def ensure_file(file_path, csv_header=None):    
    """
    确保文件路径中的文件夹和文件存在。
    - 如果文件夹不存在，则递归创建。
    - 如果文件不存在，则创建文件。
    - 如果文件已存在，则什么也不做。
    """
    # 获取文件夹路径
    folder_path = os.path.dirname(file_path)
    
    # 如果文件夹路径不为空且文件夹不存在，则递归创建
    if folder_path and not os.path.exists(folder_path):
        print(f"Creating folders: {folder_path}")
        os.makedirs(folder_path)
    
    # 如果文件不存在，则创建文件
    if not os.path.exists(file_path):
        print(f"Creating file: {file_path}")
        if csv_header is not None:
            pd.DataFrame(columns=csv_header).to_csv(file_path, index=True)
        else:
            with open(file_path, 'w') as f:
                pass  # 创建一个空文件

def ensure_dir(dir_path):    
    """
    确保文件路径中的文件夹存在。
    - 如果文件夹不存在，则递归创建。
    """
    
    # 如果文件夹路径不为空且文件夹不存在，则递归创建
    if dir_path and not os.path.exists(dir_path):
        print(f"Creating folders: {dir_path}")
        os.makedirs(dir_path)

class Trie:

    def __init__(self, seqs):
        self.root = {}
        for seq in seqs:
            self.insert(seq)
        
    
    def insert(self, seq):
        node = self.root
        for word in seq:
            if word not in node:
                node[word] = {}
            node = node[word]
    
    def select_fit_prefix(
        self,
        prefix,
    ):
        node = self.root
        if len(prefix) == 0:
            return list(node.keys())
        for word in prefix:
            if word not in node:
                return []
            node = node[word]
        return list(node.keys())

def get_prefix_allowed_tokens_fn(trie):
    def prefix_allowed_tokens_fn(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = trie.select_fit_prefix(sentence)
        return trie_out
    return prefix_allowed_tokens_fn
from copy import deepcopy
import json
import os
from typing import Literal
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
import scipy.sparse as sp

from tqdm import tqdm

from util.util import ensure_dir

class SeqRecDataset(Dataset):

    def __init__(self, 
        data_root_path, 
        dataset, 
        max_len, 
        mode:Literal["train", "valid", "test"], 
        seed=1024,
    ):
        self.data_root_path = data_root_path
        self.dataset = dataset
        self.max_len = max_len
        self.mode = mode

        self.inter_path = os.path.join(data_root_path, dataset, f"{dataset}.inter.json")
        assert os.path.exists(self.inter_path), f"Inter file not found in {self.inter_path}"

        self.inters = self._load_inter(self.inter_path)
        self.num_items = self.get_item_num()
        self.num_users = self.get_user_num()

        self.cache_dir_path = os.path.join(data_root_path, dataset, "cache/SeqRecDataset")
        ensure_dir(self.cache_dir_path)

        self.cache_file_path = os.path.join(data_root_path, dataset, f"cache/SeqRecDataset/{seed}-{max_len}-{mode}.npy")
        if os.path.exists(self.cache_file_path):
            self.inter_data = np.load(self.cache_file_path, allow_pickle=True)
            return

        if self.mode == "train":
            self.inter_data = self._process_train_data()
            np.save(self.cache_file_path, self.inter_data)
        elif self.mode == "valid":
            self.inter_data = self._process_valid_data()
            np.save(self.cache_file_path, self.inter_data)
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
            np.save(self.cache_file_path, self.inter_data)
    
    def __len__(self):
        _len = len(self.inter_data)
        return _len
    
    def __getitem__(self, idx):
        sample = self.inter_data[idx]
        return sample

    def _load_inter(self, inter_path):
        inters = json.load(open(inter_path, "r"))
        return inters

    def _process_train_data(self):
        inter_data = []
        for user_id, items in tqdm(self.inters.items()):
            items = items[:-2]
            items = [item + 1 for item in items] # 0 for padding
            for i in range(1, len(items)):
                # normal data sample
                seq = items[:i]
                target = items[i]
                if len(seq) < self.max_len:
                    seq = seq + [0] * (self.max_len - len(seq))
                else:
                    seq = seq[-self.max_len:]
                
                sample = {
                    "user_seqs": np.array(int(user_id)),
                    "his_seqs": np.array(seq),
                    "next_items": np.array(target),
                }
                inter_data.append(sample)
        return inter_data
    
    def _process_valid_data(self):
        inter_data = []
        for user_id, items in tqdm(self.inters.items()):
            items = items[:-1]
            items = [item + 1 for item in items]

            # normal data sample
            seq = items[:-1]
            target = items[-1]
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                seq = seq[-self.max_len:]

            sample = {
                "user_seqs": np.array(int(user_id)),
                "his_seqs": np.array(seq),
                "next_items": np.array(target),
            }
            inter_data.append(sample)
        return inter_data
    
    def _process_test_data(self):
        inter_data = []
        for user_id, items in tqdm(self.inters.items()):
            items = [item + 1 for item in items]

            # normal data sample
            seq = items[:-1]
            target = items[-1]
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                seq = seq[-self.max_len:]

            sample = {
                "user_seqs": np.array(int(user_id)),
                "his_seqs": np.array(seq),
                "next_items": np.array(target),
            }
            inter_data.append(sample)
        return inter_data
    
    def get_item_num(self):
        return max([max(items) for items in self.inters.values()]) + 1
    
    def get_user_num(self):
        return len(self.inters)

    def get_adjacency_matrix(self):
        cache_adj_path = os.path.join(self.cache_dir_path, "sp_adj_mat.npz")
        if os.path.exists(cache_adj_path):
            sp_adj_mat = sp.load_npz(cache_adj_path)
            return sp_adj_mat
        inter_matrix = np.array([ [int(uid), int(sid), 1] for uid, sid_list in self.inters.items() for sid in sid_list])
        inter_matrix[:, 1] += self.num_users
        inter_matrix_t = deepcopy(inter_matrix)
        inter_matrix_t[:, [0, 1]] = inter_matrix[:, [1, 0]]
        A = np.concatenate([inter_matrix, inter_matrix_t], axis=0)
        adf_mat = sp.csr_matrix(
            (A[:, 2], (A[:, 0], A[:, 1])),
            shape=(self.num_users + self.num_items, self.num_users + self.num_items)
        )
        rowsum = np.array(adf_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adf_mat).dot(d_mat_inv)
        sp.save_npz(cache_adj_path, norm_adj)
        return norm_adj

class IDDataset(Dataset):
    def __init__(self, num_items):
        self.data = np.arange(1, 1 + num_items)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Index is to compute cf embedding"""
        return self.data[index]

from collections import defaultdict
import random
from typing import List, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from k_means_constrained import KMeansConstrained
from layers.MLPLayers import MLPLayers


class VectorQuantier(nn.Module):
    
    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,

        sinkhorn_open: bool=True,
        sinkhorn_epsilon: float=3e-3,
        sinkhorn_iter: int=50,

        diversity_loss_open: bool=False,
        kmeans_init_iter: int=10,
        kmeans_cluster: int=10,

        mu: float=0.25,
        beta: float=0.0001,
    ):
        super(VectorQuantier, self).__init__()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.labels = torch.arange(codebook_size).tolist()


        self.sinkhorn_open = sinkhorn_open
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iter = sinkhorn_iter

        self.diversity_loss_open = diversity_loss_open
        self.kmeans_init_iter = kmeans_init_iter
        self.kmeans_cluster = kmeans_cluster


        self.mu = mu
        self.beta = beta
    
    def kmeans_init(self, x, sample_indices=False):
        # x: [N, codebook_dim]
        size_min = min(x.shape[0] // (self.codebook_size * 2), 50)
        x_ = x.detach().cpu().numpy()
        try:
            clf = KMeansConstrained(
                n_clusters=self.codebook_size,
                size_min=size_min, 
                size_max=size_min * 4, 
                max_iter=self.kmeans_init_iter, 
                n_init=10,
                n_jobs=10, 
                verbose=False
            )
            clf.fit(x_)
        except Exception as e:
            print("KMeansConstrained Error when kmeans_init: ", e)
            clf = KMeansConstrained(
                n_clusters=self.codebook_size,
                size_min=size_min, 
                size_max=size_min * 4, 
                max_iter=self.kmeans_init_iter, 
                n_init=10,
                n_jobs=10, 
                verbose=False
            )
            clf.fit(x_)
        self.codebook.weight.data.copy_(torch.tensor(clf.cluster_centers_, dtype=torch.float32))

        quant_hard, _, _, _ = self.inference(x, sample_indices)
        return quant_hard

    def update_labels(self):
        size_min = min(self.codebook_size // (self.kmeans_cluster * 2), 10)
        x_ = self.codebook.weight.detach().cpu().numpy()
        try:
            clf = KMeansConstrained(
                n_clusters=self.kmeans_cluster,
                size_min=size_min, 
                size_max=size_min * 6, 
                max_iter=10, 
                n_init=10,
                n_jobs=10, 
                verbose=False
            )
            clf.fit(x_)
        except Exception as e:
            print("KMeansConstrained Error when kmeans_init: ", e)
            clf = KMeansConstrained(
                n_clusters=self.kmeans_cluster,
                size_min=size_min, 
                size_max=size_min * 6, 
                max_iter=10, 
                n_init=10,
                n_jobs=10, 
                verbose=False
            )
            clf.fit(x_)
        self.labels = torch.tensor(clf.labels_, dtype=torch.long).tolist()

    def forward(self, x, sample_indices=False):
        # x [N, codebook_dim]
        d2term = (
            torch.sum(x**2, dim=-1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=-1, keepdim=True).t()
            - 2 * torch.matmul(x, self.codebook.weight.t())
        ) # [N, codebook_size]

        if sample_indices:
            if self.sinkhorn_open and self.sinkhorn_epsilon > 0:
                # Sinkhorn normalization -> [-1, 1]
                max_d = d2term.max()
                min_d = d2term.min()
                mid_d = (max_d + min_d) / 2
                d2term_normed = (d2term - mid_d) / (max_d - mid_d + 1e-5)
                d2term_normed = d2term_normed.double()

                # Sinkhorn iteration 概率均衡化
                d2term_standard = torch.exp(-d2term_normed / self.sinkhorn_epsilon)
                B = d2term_standard.shape[0]
                K = d2term_standard.shape[1] 
                d2term_standard = d2term_standard / (d2term_standard.sum(dim=1, keepdim=True).sum(dim=0, keepdim=True))

                for _ in range(self.sinkhorn_iter):
                    # row normalization
                    d2term_standard = d2term_standard / (d2term_standard.sum(dim=1, keepdim=True))
                    d2term_standard /= B

                    # column normalization
                    d2term_standard = d2term_standard / (d2term_standard.sum(dim=0, keepdim=True))
                    d2term_standard /= K


                
                d2term_standard *= B

                probs = d2term_standard # [batch_size, codebook_size]
                indices = torch.argmax(probs, dim=-1) # [batch_size]
            else:
                probs = F.softmax(-d2term, dim=-1)
                indices = torch.multinomial(probs, 1)
        else:
            probs = F.softmax(-d2term, dim=-1)
            indices = torch.argmin(d2term, dim=-1)

        quant_hard = self.codebook(indices) # [N, codebook_dim]

        codebook_loss = torch.mean((quant_hard - x.detach())**2, dim=-1) # [N]
        commitment_loss = torch.mean((quant_hard.detach() - x)**2, dim=-1) # [N]

        loss = codebook_loss + self.mu*commitment_loss # [N]

        if self.diversity_loss_open and self.beta > 0:
            # 计算每一个物品对应的cluster
            cluster_indices = [self.labels[idx.item()] for idx in indices]

            # 计算每一个cluster对应物品集进行码表对应后的term index list
            cluster_index_list = {}
            for cluster in range(self.kmeans_cluster):
                cluster_index_list[cluster] = [
                    index for index, index_cluster in enumerate(self.labels) if index_cluster == cluster
                ]

            # # 计算每一个item对应的cluster的 term index list
            # index_pos_list = [cluster_index_list[cluster] for cluster in cluster_indices]

            # # 对每一个item在同一个cluster中随机采样一个不是自己的item 对应的 term index
            # pos_sample = []
            # for idx, pos_index_list in enumerate(index_pos_list):
            #     sample = random.choice(pos_index_list)
            #     while sample == indices[idx]:
            #         sample = random.choice(pos_index_list)
            #     pos_sample.append(sample)
            
            # pos = torch.tensor(pos_sample).to(x.device)

            # sim = quant_hard @ self.codebook.weight.t() # [N, codebook_size]

            # # 防止diversity loss打击自己
            # sim_self = torch.zeros_like(sim)
            # for idx in range(sim_self.shape[0]):
            #     sim_self[idx, indices[idx]] = 1e12
            # sim = sim - sim_self

            # 采样优化，一次采样
            all_candidates = []
            candidate_ptr = []
            for idx in range(x.shape[0]):
                cluster = cluster_indices[idx]
                candidates = cluster_index_list[cluster]
                valid_candidates = [c for c in candidates if c != indices[idx].item()]
                if not valid_candidates:
                    valid_candidates = [indices[idx].item()]
                start = len(all_candidates)
                all_candidates.extend(valid_candidates)
                end = len(all_candidates)
                candidate_ptr.append((start, end))

            all_candidates = torch.tensor(all_candidates, dtype=torch.long, device=x.device)
            ptr_tensor = torch.tensor(candidate_ptr, dtype=torch.long, device=x.device)

            rand_uniform = torch.rand(x.shape[0], device=x.device)
            candidate_lens = ptr_tensor[:, 1] - ptr_tensor[:, 0]
            rand_offset = (rand_uniform * (candidate_lens.float())).long()
            pos = all_candidates[ptr_tensor[:, 0] + rand_offset]

            sim = quant_hard @ self.codebook.weight.t() # [N, codebook_size]

            sim.scatter_(1, indices.unsqueeze(1), float("-inf"))

            diversity_loss = F.cross_entropy(sim, pos, reduction="none") # [N]

            loss += self.beta * diversity_loss # [N]
        
        # 这里 quant的分离操作必须放到loss的计算之后否则会出现异常，原理暂不明晰，暂定为动态计算图问题
        quant_soft = (quant_hard - x).detach() + x # [N, codebook_dim]

        return quant_hard, quant_soft, indices, probs, loss

    @torch.no_grad()
    def inference(self, x, sample_indices=False):
        # x [N, codebook_dim]
        d2term = (
            torch.sum(x**2, dim=-1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=-1, keepdim=True).t()
            - 2 * torch.matmul(x, self.codebook.weight.t())
        )

        if sample_indices:
            if self.sinkhorn_open and self.sinkhorn_epsilon > 0:
                # Sinkhorn normalization -> [-1, 1]
                max_d = d2term.max()
                min_d = d2term.min()
                mid_d = (max_d + min_d) / 2
                d2term_normed = (d2term - mid_d) / (max_d - mid_d + 1e-5)
                d2term_normed = d2term_normed.double()

                # Sinkhorn iteration 概率均衡化
                d2term_standard = torch.exp(-d2term_normed / self.sinkhorn_epsilon)
                B = d2term_standard.shape[0]
                K = d2term_standard.shape[1] 
                d2term_standard = d2term_standard / (d2term_standard.sum(dim=1, keepdim=True).sum(dim=0, keepdim=True))

                for _ in range(self.sinkhorn_iter):
                    # row normalization
                    d2term_standard = d2term_standard / (d2term_standard.sum(dim=1, keepdim=True))
                    d2term_standard /= B

                    # column normalization
                    d2term_standard = d2term_standard / (d2term_standard.sum(dim=0, keepdim=True))
                    d2term_standard /= K
                
                d2term_standard *= B

                probs = d2term_standard # [batch_size, codebook_size]
                indices = torch.argmax(probs, dim=-1) # [batch_size]
            else:
                probs = F.softmax(-d2term, dim=-1)
                indices = torch.multinomial(probs, 1) 
        else:
            probs = F.softmax(-d2term, dim=-1)
            indices = torch.argmin(d2term, dim=-1)

        quant_hard = self.codebook(indices) # [N, codebook_dim]
        quant_soft = (quant_hard - x).detach() + x # [N, codebook_dim]

        return quant_hard, quant_soft, indices, probs
            

class ResidualVectorQuantier(nn.Module):

    def __init__(
        self,
        codebook_sizes: List[int]=[256, 256, 256, 256],
        codebook_dim: int=32,

        sinkhorn_open: bool=True,
        sinkhorn_epsilons: List[float]=[0.0, 0.0, 0.0, 0.003],
        sinkhorn_iter: int=50,

        diversity_loss_open: bool=False,
        kmeans_init_iter: int=10,
        kmeans_cluster: int=10,

        mu: float=0.25,
        beta: float=0.0001,
    ):
        super(ResidualVectorQuantier, self).__init__()

        self.codebook_sizes = codebook_sizes
        self.codebook_dim = codebook_dim

        self.sinkhorn_open = sinkhorn_open
        self.sinkhorn_epsilons = sinkhorn_epsilons
        self.sinkhorn_iter = sinkhorn_iter

        self.diversity_loss_open = diversity_loss_open
        self.kmeans_init_iter = kmeans_init_iter
        self.kmeans_cluster = kmeans_cluster

        self.mu = mu
        self.beta = beta

        self.codebooks = nn.ModuleList(
            VectorQuantier(
                codebook_size=codebook_size_i,
                codebook_dim=self.codebook_dim,
                sinkhorn_open=self.sinkhorn_open,
                sinkhorn_epsilon=sinkhorn_epsilon_i,
                sinkhorn_iter=self.sinkhorn_iter,
                diversity_loss_open=self.diversity_loss_open,
                kmeans_init_iter=self.kmeans_init_iter,
                kmeans_cluster=self.kmeans_cluster,
                mu=self.mu,
                beta=self.beta
            )
            for codebook_size_i, sinkhorn_epsilon_i in zip(self.codebook_sizes, self.sinkhorn_epsilons)
        )

    def kmeans_init(self, x):
        # x: [N, codebook_dim]
        residual = x
        for idx, codebook_i in enumerate(self.codebooks):
            if idx == len(self.codebooks) - 1:
                x_q = codebook_i.kmeans_init(residual, True)
            else:
                x_q = codebook_i.kmeans_init(residual, False)
            residual = residual - x_q

    def update_labels(self):
        for codebook_i in self.codebooks:
            codebook_i.update_labels()

    def forward(self, x):
        # x [N, codebook_dim]
        residual = x
        quant_hard = 0
        quant_soft = 0
        indices_list = []
        probs_list = []
        loss_list = []
        for idx, codebook_i in enumerate(self.codebooks):
            if idx == len(self.codebooks) - 1:
                quant_hard_i, quant_soft_i, indices_i, probs_i, loss_i = codebook_i(residual, True)  
            else:   
                quant_hard_i, quant_soft_i, indices_i, probs_i, loss_i = codebook_i(residual, False)
            indices_list.append(indices_i)
            probs_list.append(probs_i)
            loss_list.append(loss_i)
            residual = residual - quant_soft_i
            quant_hard += quant_hard_i
            quant_soft += quant_soft_i
        
        indices = torch.stack(indices_list, dim=-1) # [N, codebook_num]
        probs = torch.stack(probs_list, dim=-1) # [N, codebook_size, codebook_num]
        loss = torch.stack(loss_list, dim=-1).mean(dim=-1) # [N]

        return quant_hard, quant_soft, indices, probs, loss

    @torch.no_grad()
    def inference(self, x):
        # x [N, codebook_dim]
        residual = x
        quant_hard = 0
        quant_soft = 0
        indices_list = []
        probs_list = []
        for idx, codebook_i in enumerate(self.codebooks):
            if idx == len(self.codebooks) - 1:
                quant_hard_i, quant_soft_i, indices_i, probs_i = codebook_i.inference(residual, True)
            else:
                quant_hard_i, quant_soft_i, indices_i, probs_i = codebook_i.inference(residual, False)
            indices_list.append(indices_i)
            probs_list.append(probs_i)
            residual = residual - quant_soft_i
            quant_hard += quant_hard_i
            quant_soft += quant_soft_i
        
        indices = torch.stack(indices_list, dim=-1) # [N, codebook_num]
        probs = torch.stack(probs_list, dim=-1) # [N, codebook_size, codebook_num]

        return quant_hard, quant_soft, indices, probs

    @torch.no_grad()
    def generate_indices(self, x):
        # x [N, codebook_dim]
        residual = x
        indices_list = []
        for codebook_i in self.codebooks:
            _, quant_soft_i, indices_i, _ = codebook_i.inference(residual, False)
            indices_list.append(indices_i)
            residual = residual - quant_soft_i
        
        indices = torch.stack(indices_list, dim=-1) # [N, codebook_num]

        return indices


class RQVAE(nn.Module):

    def __init__(
        self,
        item_emb_path: str="data/Beauty/Beauty.emb-llama-td.npy",
        cf_emb_path: str="data/Beauty/Beauty-cf_emb.pt",

        in_dims: List[int]=[4096, 2048, 1024, 512, 256, 128, 64, 32],
        codebook_dim: int=32,
        codebook_sizes: List[int]=[256, 256, 256, 256],
        dropout: float=0,

        sinkhorn_open: bool=True,
        sinkhorn_epsilons: List[float]=[0.0, 0.0, 0.0, 0.003],
        sinkhorn_iter: int=50,

        cf_loss_open: bool=False,
        diversity_loss_open: bool=False,
        kmeans_init_iter: int=10,
        kmeans_cluster: int=10,

        mu: float=0.25,
        alpha: float=0.02,
        beta: float=0.0001,
    ):
        
        super(RQVAE, self).__init__()
        self.in_dims = in_dims
        self.codebook_dim = codebook_dim
        self.codebook_sizes = codebook_sizes
        self.dropout = dropout

        self.sinkhorn_open = sinkhorn_open
        self.sinkhorn_epsilons = sinkhorn_epsilons
        self.sinkhorn_iter = sinkhorn_iter

        self.cf_loss_open = cf_loss_open
        self.diversity_loss_open = diversity_loss_open
        self.kmeans_init_iter = kmeans_init_iter
        self.kmeans_cluster = kmeans_cluster

        self.mu = mu
        self.alpha = alpha
        self.beta = beta

        self.encoder = MLPLayers(in_dims, activation_fn="relu", last_activation=False, dropout=dropout)
        self.rq = ResidualVectorQuantier(
            codebook_sizes=self.codebook_sizes,
            codebook_dim=self.codebook_dim,
            sinkhorn_open=self.sinkhorn_open,
            sinkhorn_epsilons=self.sinkhorn_epsilons,
            sinkhorn_iter=self.sinkhorn_iter,
            diversity_loss_open=self.diversity_loss_open,
            kmeans_init_iter=self.kmeans_init_iter,
            kmeans_cluster=self.kmeans_cluster,
            mu=self.mu,
            beta=self.beta
        )
        self.decoder = MLPLayers(in_dims[::-1], activation_fn="relu", last_activation=False, dropout=dropout)


        self.padding_idx = 0
        self.item_emb_data = np.load(item_emb_path)
        self.item_emb_data = np.concatenate([np.zeros((1, self.item_emb_data.shape[1])), self.item_emb_data], axis=0)
        self.item_emb = nn.Embedding(self.item_emb_data.shape[0], self.in_dims[0], padding_idx=self.padding_idx)
        self.item_emb.weight.data.copy_(torch.tensor(self.item_emb_data, dtype=torch.float32))
        self.item_emb.weight.requires_grad = False

        if self.cf_loss_open:
            self.cf_emb_data = torch.cat(
                [
                    torch.zeros((1, self.in_dims[-1]), dtype=torch.long),
                    torch.load(cf_emb_path)
                ],
                dim=0
            )
            self.cf_emb = nn.Embedding(self.cf_emb_data.shape[0], self.in_dims[-1], padding_idx=self.padding_idx)
            self.cf_emb.weight.data.copy_(self.cf_emb_data.clone().detach().to(torch.float32))
            self.cf_emb.weight.requires_grad = False

            assert self.item_emb.weight.shape[0] == self.cf_emb.weight.shape[0], "cf emb or llm emb have wrong shape"

        self.item_indices = None

    @torch.no_grad()
    def kmeans_init(self):
        x_in = self.encoder(self.item_emb.weight[1:]) # [N, in_dims[-1]]
        self.rq.kmeans_init(x_in)

    @torch.no_grad()
    def update_labels(self):
        self.rq.update_labels()

    @torch.no_grad()
    def compute_collision_ratio(self):
        x_in = self.encoder(self.item_emb.weight[1:]) # [N, in_dims[-1]]
        item_indices = self.rq.generate_indices(x_in).cpu().numpy()
        inverse_map = defaultdict(list)
        for i, indices in enumerate(item_indices):
            key = ",".join(map(str, indices))
            inverse_map[key].append(i)
        print(
            f"Unique key number: {len(inverse_map)}, collision ratio: {(len(item_indices) - len(inverse_map))/len(item_indices)}."
        )
        return (len(item_indices) - len(inverse_map))/len(item_indices)

    @torch.no_grad()
    def get_all_indices(self):
        return self.item_indices
    
    def set_all_indices(self):

        def get_collision_item(all_indices_str):
            indexstr2id = {}
            for i, indexstr in enumerate(all_indices_str):
                if indexstr not in indexstr2id:
                    indexstr2id[indexstr] = []
                indexstr2id[indexstr].append(i)

            collision_item_groups = []

            for index in indexstr2id:
                if len(indexstr2id[index]) > 1:
                    collision_item_groups.append(indexstr2id[index])

            return collision_item_groups



        x_in = self.encoder(self.item_emb.weight[1:])
        item_indices = self.rq.generate_indices(x_in).cpu().numpy() # [N, codebook_num]
        prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>","<f_{}>"]
        item_indices_str = []
        for indices in item_indices:
            item_indices_str.append(",".join([prefix[j].format(indices[j]) for j in range(len(indices))]))
        
        sinkhorn_open = self.sinkhorn_open
        sinkhorn_epsilons = self.sinkhorn_epsilons

        self.sinkhorn_open = True
        self.sinkhorn_epsilons = [0.0, 0.0, 0.0, 0.003]
        self.rq.sinkhorn_open = True
        self.rq.sinkhorn_epsilons = [0.0, 0.0, 0.0, 0.003]
        for i, vq in  enumerate(self.rq.codebooks):
            vq.sinkhorn_open = True
            vq.sinkhorn_epsilon = self.sinkhorn_epsilons[i]
        
        iter_num = 20
        while iter_num > 0 and len(item_indices) != len(set(item_indices_str)):
            collision_item_groups = get_collision_item(item_indices_str)
            for collision_group in collision_item_groups:
                _, _, new_indices, _ = self.inference(torch.tensor(collision_group, dtype=torch.long, device=x_in.device) + 1)
                new_indices = new_indices.cpu().numpy()
                for i, new_index in enumerate(new_indices):
                    item_indices_str[collision_group[i]] = ",".join([prefix[j].format(new_index[j]) for j in range(len(new_index))])
                    item_indices[collision_group[i]] = new_index
            iter_num -= 1
        
        self.sinkhorn_open = sinkhorn_open
        self.sinkhorn_epsilons = sinkhorn_epsilons
        self.rq.sinkhorn_open = sinkhorn_open
        self.rq.sinkhorn_epsilons = sinkhorn_epsilons
        for i, vq in  enumerate(self.rq.codebooks):
            vq.sinkhorn_open = sinkhorn_open
            vq.sinkhorn_epsilon = sinkhorn_epsilons[i]
        
        self.item_indices = torch.cat(
            [
                torch.zeros((1, 4), dtype=torch.long, device=x_in.device),
                torch.tensor(item_indices, dtype=torch.long, device=x_in.device)
            ]
        )
        
        print(
            f"Unique key number: {len(set(item_indices_str))}, collision ratio: {(len(item_indices) - len(set(item_indices_str)))/len(item_indices)}."
        )

    def forward(self, x):
        # x: (batch_size) or (batch_size, seq_len)
        shape_len = len(x.shape)
        assert shape_len == 1 or shape_len == 2, "x shape must be (batch_size) or (batch_size, seq_len)"
        if shape_len == 2:
            bsz, seq_len = x.shape
            x_reshape = x.reshape(-1)
        else:
            x_reshape = x

        mask = (x_reshape != self.padding_idx).float() # [N]

        x_emb = self.item_emb(x_reshape) # [N, in_dims[0]]
        x_in = self.encoder(x_emb) # [N, in_dims[-1]]
        quant_hard, quant_soft, indices, probs, quant_loss = self.rq(x_in)
        x_out = self.decoder(quant_soft) # [N, in_dims[0]]

        recon_loss = torch.mean((x_out - x_emb)**2, dim=-1) * mask
        recon_loss = torch.sum(recon_loss) / torch.sum(mask)

        quant_loss = quant_loss * mask
        quant_loss = torch.sum(quant_loss) / torch.sum(mask)

        loss = recon_loss + quant_loss

        if self.cf_loss_open and self.alpha > 0:
            x_cf_emb = self.cf_emb(x_reshape) # [N, in_dims[-1]]
            sim = quant_soft @ x_cf_emb.t() # [N, N]
            label = torch.arange(x_reshape.shape[0], device=x_reshape.device)
            cf_loss = F.cross_entropy(sim, label, reduction="none")
            cf_loss = torch.sum(cf_loss * mask) / torch.sum(mask)
            loss += self.alpha * cf_loss
        else:
            cf_loss = torch.tensor(0.0)
            
        if shape_len == 2:
            quant_hard = quant_hard.reshape(bsz, seq_len, -1) # [bsz, seq_len, codebook_dim]
            quant_soft = quant_soft.reshape(bsz, seq_len, -1) # [bsz, seq_len, codebook_dim]
            indices = indices.reshape(bsz, seq_len, -1) # [bsz, seq_len, codebook_num]
            probs = probs.reshape(bsz, seq_len, probs.shape[-2], probs.shape[-1]) # [bsz, seq_len, codebook_size, codebook_num]
        return quant_hard, quant_soft, indices, probs, loss, recon_loss, quant_loss, cf_loss
    
    @torch.no_grad()
    def inference(self, x):
        # x: (batch_size) or (batch_size, seq_len)
        shape_len = len(x.shape)
        assert shape_len == 1 or shape_len == 2, "x shape must be (batch_size) or (batch_size, seq_len)"
        if shape_len == 2:
            bsz, seq_len = x.shape
            x = x.reshape(-1)

        x_emb = self.item_emb(x) # [N, in_dims[0]]
        x_in = self.encoder(x_emb) # [N, in_dims[-1]]
        quant_hard, quant_soft, indices, probs = self.rq.inference(x_in)
        
        if shape_len == 2:
            quant_hard = quant_hard.reshape(bsz, seq_len, -1) # [bsz, seq_len, codebook_dim]
            quant_soft = quant_soft.reshape(bsz, seq_len, -1) # [bsz, seq_len, codebook_dim]
            indices = indices.reshape(bsz, seq_len, -1) # [bsz, seq_len, codebook_num]
            probs = probs.reshape(bsz, seq_len, probs.shape[-2], probs.shape[-1]) # [bsz, seq_len, codebook_size, codebook_num]

        return quant_hard, quant_soft, indices, probs

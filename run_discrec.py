
import argparse
import math
import os
import random
import time
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
from torch import nn

from data.dataset import IDDataset, SeqRecDataset
from layers.DiscRec import DiscRec
from transformers.models.t5.configuration_t5 import T5Config
from layers.Rqvae import RQVAE
from torch.optim.lr_scheduler import LambdaLR

from util.evaluate import ndcg_at_k, recall_at_k
from util.util import ensure_dir, ensure_file


MODEL_NAME="DiscRec"

def parser_args():
    parser = argparse.ArgumentParser(description=MODEL_NAME)

    # global
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")

    # data
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--dataset", choices=["Beauty", "Arts", "Instruments", "Toys"], default="Beauty")
    parser.add_argument("--num_workers", type=int, default=4)

    # rqvae
    parser.add_argument("--id_type", choices=["tiger", "letter"], default="tiger")
    parser.add_argument("--in_dims", type=list, default=[4096, 2048, 1024, 512, 256, 128, 64, 32])
    parser.add_argument("--codebook_dim", type=int, default=32)
    parser.add_argument("--codebook_sizes", type=list, default=[256, 256, 256, 256])
    parser.add_argument("--rqvae_dropout", type=float, default=0)
    parser.add_argument("--sinkhorn_open", action="store_true")
    parser.add_argument("--sinkhorn_epsilons", type=list, default=[0.0, 0.0, 0.0, 0.003])
    parser.add_argument("--sinkhorn_iter", type=int, default=50)
    parser.add_argument("--kmeans_init_iter", type=int, default=10)
    parser.add_argument("--kmeans_cluster", type=int, default=10)
    parser.add_argument("--mu", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--beta", type=float, default=0.0001)

    # rqvae train and eval
    parser.add_argument("--rqvae_epochs", type=int, default=20000)
    parser.add_argument("--rqvae_batch_size", type=int, default=1024)
    parser.add_argument("--rqvae_lr", type=float, default=1e-3)
    parser.add_argument("--rqvae_wd", type=float, default=1e-4)
    parser.add_argument("--rqvae_optimizer", choices=["adamw"], default="adamw")
    parser.add_argument("--rqvae_eval_step", type=int, default=2000)
    parser.add_argument("--rqvae_eval_metric", choices=["unique_key_ratio"], default="unique_key_ratio")

    # t54rec 
    parser.add_argument("--t54rec_config_dir_path", type=str, default="./extra_config/")
    parser.add_argument("--rqvae_select_position", choices=["last", "best"], default="last")

    # t54rec train and eval
    parser.add_argument("--t54rec_epochs", type=int, default=200)
    parser.add_argument("--t54rec_train_batch_size", type=int, default=256)
    parser.add_argument("--t54rec_valid_batch_size", type=int, default=64)
    parser.add_argument("--t54rec_test_batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--t54rec_lr", type=float, default=5e-4)
    parser.add_argument("--t54rec_wd", type=float, default=1e-2)
    parser.add_argument("--t54rec_optimizer", choices=["adamw"], default="adamw")
    parser.add_argument("--scheduler_type", choices=["cosine", "linear", "none"], default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--t54rec_eval_step", type=int, default=1)
    parser.add_argument("--t54rec_early_stop_step", type=int, default=20)
    parser.add_argument("--t54rec_eval_metric", choices=["loss"], default="loss")
    parser.add_argument("--beam_size", type=int, default=20)

    # test
    parser.add_argument("--metrics", nargs="+", choices=["Recall", "NDCG"], default=["Recall", "NDCG"])
    parser.add_argument("--topk", nargs="+", type=int, default=[5, 10])

    # log, save and result
    parser.add_argument("--save_root_path", type=str, default="./save/")

    return parser.parse_args()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(args):
    return torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

def initial_dataLoader(args):

    datasets = {
        "train": SeqRecDataset(
            data_root_path=args.data_path, 
            dataset=args.dataset, 
            max_len=args.max_len, 
            mode="train", 
            seed=args.seed
        ),
        "valid": SeqRecDataset(
            data_root_path=args.data_path, 
            dataset=args.dataset, 
            max_len=args.max_len, 
            mode="valid", 
            seed=args.seed
        ),
        "test": SeqRecDataset(
            data_root_path=args.data_path, 
            dataset=args.dataset, 
            max_len=args.max_len, 
            mode="test", 
            seed=args.seed
        )
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"], 
            batch_size=args.t54rec_train_batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        ),
        "valid": DataLoader(
            datasets["valid"], 
            batch_size=args.t54rec_valid_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        ),
        "test": DataLoader(
            datasets["test"], 
            batch_size=args.t54rec_test_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    }

    return dataloaders["train"], dataloaders["valid"], dataloaders["test"], datasets["train"].num_items, datasets["train"].num_users

def initial_model(args, device):
    if args.id_type == "tiger":
        rqvae_state_path = f"{args.save_root_path}/{args.dataset}/{MODEL_NAME}/rqvae-tiger-lr_{args.rqvae_lr}-wd_{args.rqvae_wd}-kmi_{args.kmeans_init_iter}-{args.rqvae_select_position}.pth"
    else:
        rqvae_state_path = f"{args.save_root_path}/{args.dataset}/{MODEL_NAME}/rqvae-letter-alpha_{args.alpha}-beta_{args.beta}-lr_{args.rqvae_lr}-wd_{args.rqvae_wd}-kmi_{args.kmeans_init_iter}-{args.rqvae_select_position}.pth"
    if not os.path.exists(rqvae_state_path):
        if args.id_type == "tiger":
            rqvae = RQVAE(
                item_emb_path=f"{args.data_path}/{args.dataset}/{args.dataset}.emb-llama-td.npy",
                in_dims=args.in_dims,
                codebook_dim=args.codebook_dim,
                codebook_sizes=args.codebook_sizes,
                dropout=args.rqvae_dropout,
                sinkhorn_open=args.sinkhorn_open,
                sinkhorn_epsilons=args.sinkhorn_epsilons,
                sinkhorn_iter=args.sinkhorn_iter,
                cf_loss_open=False,
                diversity_loss_open=False,
                kmeans_init_iter=args.kmeans_init_iter,
                mu=args.mu
            ).to(device)
        else:
            rqvae = RQVAE(
                item_emb_path=f"{args.data_path}/{args.dataset}/{args.dataset}.emb-llama-td.npy",
                cf_emb_path=f"{args.data_path}/{args.dataset}/{args.dataset}-cf_emb.pt",
                in_dims=args.in_dims,
                codebook_dim=args.codebook_dim,
                codebook_sizes=args.codebook_sizes,
                dropout=args.rqvae_dropout,
                sinkhorn_open=args.sinkhorn_open,
                sinkhorn_epsilons=args.sinkhorn_epsilons,
                sinkhorn_iter=args.sinkhorn_iter,
                cf_loss_open=True,
                diversity_loss_open=True,
                kmeans_init_iter=args.kmeans_init_iter,
                kmeans_cluster=args.kmeans_cluster,
                mu=args.mu,
                alpha=args.alpha,
                beta=args.beta
            ).to(device)
        fit_rqvae(args, rqvae, device)

    assert os.path.exists(rqvae_state_path)
    rqvae = torch.load(rqvae_state_path, weights_only=False, map_location=device)

    # freeze rqvae
    for param in rqvae.parameters():
        param.requires_grad = False


    config = T5Config.from_pretrained(args.t54rec_config_dir_path)
    config.vocab_size = 256 * 4 + 2
    t54rec = DiscRec(config).to(device)

    return rqvae, t54rec

def initial_optimizer_scheduler(
    model, 
    lr, 
    wd, 
    optimizer, 
    epochs, 
    warmup_ratio, 
    batchnum_per_epoch, 
    scheduler_type
):
    # optimizer 
    if optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd
        )
    else:
        raise ValueError("Invalid optimizer.")
    
    # scheduler
    total_steps = epochs * batchnum_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda_cos(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    def lr_lambda_linear(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_lambda_cos if scheduler_type == "cosine" else lr_lambda_linear
    ) if scheduler_type != "none" else None

    return optimizer, scheduler

def fit_rqvae(
    args,
    rqvae,
    device
):
    print("Fitting RQVAE")

    save_dir_path = f"{args.save_root_path}/{args.dataset}/{MODEL_NAME}"
    ensure_dir(save_dir_path)
    if args.id_type == "letter":
        save_best_file_name = f"{save_dir_path}/rqvae-letter-alpha_{args.alpha}-beta_{args.beta}-lr_{args.rqvae_lr}-wd_{args.rqvae_wd}-kmi_{args.kmeans_init_iter}-best.pth"
        save_last_file_name = f"{save_dir_path}/rqvae-letter-alpha_{args.alpha}-beta_{args.beta}-lr_{args.rqvae_lr}-wd_{args.rqvae_wd}-kmi_{args.kmeans_init_iter}-last.pth"
    else:
        save_best_file_name = f"{save_dir_path}/rqvae-tiger-lr_{args.rqvae_lr}-wd_{args.rqvae_wd}-kmi_{args.kmeans_init_iter}-best.pth"
        save_last_file_name = f"{save_dir_path}/rqvae-tiger-lr_{args.rqvae_lr}-wd_{args.rqvae_wd}-kmi_{args.kmeans_init_iter}-last.pth"
     
    id_dataset = IDDataset(args.num_items)
    
    id_dataloader = DataLoader(
        id_dataset,
        batch_size=args.rqvae_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    optimizer, _ = initial_optimizer_scheduler(
        rqvae,
        args.rqvae_lr,
        args.rqvae_wd,
        args.rqvae_optimizer,
        args.rqvae_epochs,
        args.warmup_ratio,
        len(id_dataloader),
        "none"
    )
    rqvae.kmeans_init()

    rqvae.train()

    min_collision_ratio = 1.0
    for epoch in range(args.rqvae_epochs):
        if args.id_type == "letter":
            rqvae.update_labels()
        loss_list = []
        recon_loss_list = []
        quant_loss_list = []
        cf_loss_list = []
        for _, batch in enumerate(id_dataloader):
            batch = batch.to(device)
            _, _, _, _, loss, recon_loss, quant_loss, cf_loss = rqvae(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            recon_loss_list.append(recon_loss.item())
            quant_loss_list.append(quant_loss.item())
            cf_loss_list.append(cf_loss.item())

        if args.id_type == "letter":
            print(f"Epoch: [{epoch + 1}/{args.rqvae_epochs}], Loss: {np.mean(loss_list)}, Recon Loss: {np.mean(recon_loss_list)}, Quant Loss: {np.mean(quant_loss_list)}, CF loss: {np.mean(cf_loss_list)}")
        else:
            print(f"Epoch: [{epoch + 1}/{args.rqvae_epochs}], Loss: {np.mean(loss_list)}, Recon Loss: {np.mean(recon_loss_list)}, Quant Loss: {np.mean(quant_loss_list)}")

        if (epoch + 1) % args.rqvae_eval_step == 0:
            print(f"Evaluating Epoch: [{epoch + 1}/{args.rqvae_epochs}]")
            collision_ratio = rqvae.compute_collision_ratio()
            print(f"Epoch: [{epoch + 1}/{args.rqvae_epochs}], Collision Rate: {collision_ratio}")
            if collision_ratio <= min_collision_ratio:
                min_collision_ratio = collision_ratio
                torch.save(rqvae, save_best_file_name)
    
    # save last
    rqvae.set_all_indices()
    torch.save(rqvae, save_last_file_name)
    # save best
    rqvae = torch.load(save_best_file_name, weights_only=False, map_location=device)
    rqvae.set_all_indices()
    torch.save(rqvae,save_best_file_name)

def prepare_input(
    his_seqs,
    next_items,
    indice_matrix,
):
    # his_seqs [batch_size, max_len]
    # next_items [batch_size]

    encoder_input_ids = indice_matrix(his_seqs) # [batch_size, max_len, codebook_num]
    decoder_input_ids = indice_matrix(next_items) # [batch_size, codebook_num]

    # rescale his_seqs and next_items
    encoder_shift = (
        torch.tensor([2 + i * 256 for i in range(encoder_input_ids.shape[-1])])
        .reshape(1, 1, encoder_input_ids.shape[-1])
        .to(encoder_input_ids.device)
    )
    shift = decoder_shift = (
        torch.tensor([2 + i * 256 for i in range(decoder_input_ids.shape[-1])])
        .reshape(1, decoder_input_ids.shape[-1])
        .to(decoder_input_ids.device)
    )
    encoder_input_ids = (encoder_input_ids + encoder_shift).long() # [batch_size, max_len, codebook_num]
    decoder_input_ids = (decoder_input_ids + decoder_shift).long() # [batch_size, codebook_num]


    # write pad to encoder_input_ids
    pad_indices = (his_seqs == 0).long().nonzero(as_tuple=True)
    encoder_input_ids[pad_indices[0], pad_indices[1]] = torch.zeros(encoder_input_ids.shape[-1], dtype=torch.long).to(encoder_input_ids.device)
    encoder_input_ids = encoder_input_ids.reshape(encoder_input_ids.shape[0], -1) # [batch_size, max_len*codebook_num]

    # add eos to encoder_input_ids
    encoder_input_ids = torch.cat(
        [
            encoder_input_ids, 
            torch.ones((encoder_input_ids.shape[0], 1), dtype=torch.long).to(encoder_input_ids.device)
        ], 
        dim=-1
    ) # [batch_size, max_len*codebook_num + 1]

    # add eos to labels
    labels = torch.cat(
        [
            decoder_input_ids,
            torch.ones((decoder_input_ids.shape[0], 1), dtype=torch.long).to(encoder_input_ids.device),
        ],
        dim=-1
    ) # [batch_size, codebook_num + 1]

    # add decoder_input_token to decoder_input_ids
    decoder_input_ids = torch.cat(
        [
            torch.zeros((decoder_input_ids.shape[0], 1), dtype=torch.long).to(encoder_input_ids.device),
            decoder_input_ids
        ],
        dim=-1
    ) # [batch_size, 1 + codebook_num]

    encoder_mask = (encoder_input_ids != 0).int() # [batch_size, max_len*codebook_num + 1]

    return encoder_input_ids, encoder_mask, decoder_input_ids, labels, shift

def train_epoch(
    epoch, 
    model, 
    train_loader,
    indice_matrix,
    device, 
    optimizer, 
    scheduler
):
    
    model.train()
    total_loss = []
    print(f"Epoch [{epoch + 1}] - Start Training")
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        his_seqs = batch["his_seqs"]
        next_items = batch["next_items"]

        (
            encoder_input_ids, 
            encoder_mask, 
            decoder_input_ids, 
            labels,
            shift
        ) = prepare_input(
            his_seqs,
            next_items,
            indice_matrix
        )


        loss = model._forward(
            input_ids=encoder_input_ids,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss.append(loss.item())

        if (step + 1) % 10 == 0:
            if scheduler is not None:
                print(
                    f"Step [{step + 1}/{len(train_loader)}] - "
                    f"Avg Loss: {np.mean(total_loss):.4f}, "
                    f"Current lr: {scheduler.get_last_lr()[0]:.10f}"
                )
            else:
                print(
                    f"Step [{step + 1}/{len(train_loader)}] - "
                    f"Avg Loss: {np.mean(total_loss):.4f}"
                )
    
    print(f"Epoch [{epoch + 1}] - Avg Loss: {np.mean(total_loss):.4f}")
            
@torch.no_grad()
def eval_epoch(
    epoch, 
    model, 
    eval_loader,
    indice_matrix,
    device, 
    eval_metirc,
):
    assert eval_metirc == "loss"
    model.eval()
    total_loss = []
    print(f"Epoch [{epoch + 1}] - Start Eval")
    for step, batch in enumerate(eval_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        his_seqs = batch["his_seqs"]
        next_items = batch["next_items"]

        (
            encoder_input_ids, 
            encoder_mask, 
            decoder_input_ids, 
            labels,
            shift
        ) = prepare_input(
            his_seqs,
            next_items,
            indice_matrix
        )


        loss = model._forward(
            input_ids=encoder_input_ids,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        total_loss.append(loss.item())
    
    print(f"Eval - Avg Loss: {np.mean(total_loss):.4f}")
    return np.mean(total_loss)

@torch.no_grad()
def test(
    model,
    test_loader,
    indice_matrix,
    device,
    args,
):
    model.eval()
    print("Start Testing")

    result = {
        metric: {k: [] for k in args.topk}
        for metric in args.metrics
    }
    data_num = 0
    for step, batch in enumerate(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        his_seqs = batch["his_seqs"]
        next_items = batch["next_items"]

        (
            encoder_input_ids, 
            encoder_mask, 
            _, 
            labels,
            shift,
        ) = prepare_input(
            his_seqs,
            next_items,
            indice_matrix
        )

        item_indices = (indice_matrix.weight[1:] + shift).long().cpu().numpy()
        output_ids = model._inference(
            input_ids=encoder_input_ids,
            attention_mask=encoder_mask,
            item_indices=item_indices,
            beam_size=args.beam_size
        )
        output_ids = output_ids.cpu().numpy()
        tgt_ids = labels[:, :-1].cpu().numpy()

        tgt = np.array([".".join(map(str, x)) for x in tgt_ids])
        pred = np.array(
            [
                [".".join(map(str, x)) for x in sample]
                for sample in output_ids
            ]
        )

        for metric in args.metrics:
            if metric == "Recall":
                for k in args.topk:
                    result[metric][k].append(
                        recall_at_k(pred, tgt, k) * batch["next_items"].shape[0]
                    )
            elif metric == "NDCG":
                for k in args.topk:
                    result[metric][k].append(
                        ndcg_at_k(pred, tgt, k) * batch["next_items"].shape[0]
                    )
            else:
                raise ValueError("Invalid metric.")
        data_num += batch["next_items"].shape[0]
    
    for metric in args.metrics:
        for k in args.topk:
            result[metric][k] = np.sum(result[metric][k]) / data_num
            print(f"{metric}@{k}: {result[metric][k]:.4f}")
    
    return result


def run():
    args = parser_args()
    args.t54rec_config_dir_path = args.t54rec_config_dir_path + args.id_type
    # set seed
    set_seed(args)

    # set device
    device = get_device(args)

    # initial dataLoader
    train_loader, valid_loader, test_loader, num_items, num_users = initial_dataLoader(args)
    args.num_items = num_items
    args.num_users = num_users



    # initial model
    rqvae, t54rec = initial_model(args, device)


    for k, v in args.__dict__.items():
        print(f"{k}: {v}")

    rqvae_indices = rqvae.get_all_indices()
    indice_matrix = nn.Embedding(num_items + 1, len(args.codebook_sizes)).to(device)
    indice_matrix.weight.data.copy_(rqvae_indices)
    for param in indice_matrix.parameters():
        param.requires_grad = False

    # initial optimizer and scheduler
    optimizer, scheduler = initial_optimizer_scheduler(
        t54rec,
        args.t54rec_lr,
        args.t54rec_wd,
        args.t54rec_optimizer,
        args.t54rec_epochs,
        args.warmup_ratio,
        len(train_loader),
        args.scheduler_type
    )

    # ensure the log, save and result path
    time_now = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    args.time = time_now
    save_dir_path = os.path.join(args.save_root_path, args.dataset, MODEL_NAME)
    save_file_path = os.path.join(save_dir_path, f"{args.id_type}-DiscRec.pth")

    ensure_dir(save_dir_path)

    best_valid_metric = math.inf if args.t54rec_eval_metric == "loss" else math.inf * -1
    best_epoch = -1
    patience = 0

    for epoch in range(args.t54rec_epochs):
        train_epoch(epoch, t54rec, train_loader, indice_matrix, device, optimizer, scheduler)
        if epoch % args.t54rec_eval_step == 0:
            valid_metric = eval_epoch(epoch, t54rec, valid_loader, indice_matrix, device, args.t54rec_eval_metric)
            if args.t54rec_eval_metric == "loss":
                if valid_metric < best_valid_metric:
                    patience = 0
                    best_valid_metric = valid_metric
                    best_epoch = epoch
                    torch.save(t54rec, save_file_path)
                    print(f"Save model at epoch [{epoch + 1}]")
                else:
                    patience += 1
                    print(f"Patience: {patience}/{args.t54rec_early_stop_step}")
                    if patience >= args.t54rec_early_stop_step:
                        print(f"Early stop at epoch [{epoch + 1}]")
                        break
            else:
                raise ValueError("Invalid eval metric.")
            
    print(f"Best epoch: {best_epoch + 1}, Best valid {args.t54rec_eval_metric}: {best_valid_metric:.4f}")

    t54rec = torch.load(save_file_path, weights_only=False, map_location=device)
    test_metric = test(t54rec, test_loader, indice_matrix, device, args)


if __name__ == "__main__":
    run()
# -*- coding: UTF-8 -*-
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from Dataset import AffinityDataset
import torch.optim as optim
from torch_geometric.data import Batch
import logging
import sys
from Utils import evaluate_metric, evaluate
from module import MuLAAIP

def train_epoch(model, device, dataloader, loss_fn, optimizer, param_l2_coef, adj_loss_coef, embedding):
    model.train()
    losses = 0.0
    y_true = []
    y_pred = []
    for batch in dataloader:
        batch_ab = batch
        batch_ag = Batch.from_data_list(batch.antigen)

        embedding_ab = embedding[batch_ab.label.numpy().tolist(), :1280]
        embedding_ag = embedding[batch_ag.label.numpy().tolist(), 1280:]

        batch_ab = batch_ab.to(device)
        batch_ag = batch_ag.to(device)
        embedding_ab = torch.from_numpy(embedding_ab).to(device)
        embedding_ag = torch.from_numpy(embedding_ag).to(device)

        optimizer.zero_grad()

        output, ab_adj_mat, ag_adj_mat = model(batch_ab=batch_ab, batch_ag=batch_ag, seq_emb_ab=embedding_ab,
                                               seq_emb_ag=embedding_ag)

        loss = loss_fn(output.squeeze().to(torch.float32), batch.y.to(torch.float32))
        losses += loss.item()

        param_l2_loss = 0
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param_l2_loss += torch.norm(param, p=2)
        param_l2_loss = param_l2_coef * param_l2_loss
        adj_loss = adj_loss_coef * torch.norm(ab_adj_mat) + adj_loss_coef * torch.norm(ag_adj_mat)
        loss = loss + adj_loss + param_l2_loss

        y_true.append(batch.y.detach().cpu().numpy())
        y_pred.append(output.detach().cpu().numpy())

        loss.backward()
        optimizer.step()

    return losses, np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0).reshape(-1)


def valid_epoch(model, device, dataloader, loss_fn, embedding):
    model.eval()
    losses = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            batch_ab = batch
            batch_ag = Batch.from_data_list(batch.antigen)

            embedding_ab = embedding[batch_ab.label.numpy().tolist(), :1280]
            embedding_ag = embedding[batch_ag.label.numpy().tolist(), 1280:]

            batch_ab = batch_ab.to(device)
            batch_ag = batch_ag.to(device)
            embedding_ab = torch.from_numpy(embedding_ab).to(device)
            embedding_ag = torch.from_numpy(embedding_ag).to(device)

            output, ab_adj_mat, ag_adj_mat = model(batch_ab=batch_ab, batch_ag=batch_ag, seq_emb_ab=embedding_ab,
                                                   seq_emb_ag=embedding_ag)

            loss = loss_fn(output.squeeze().to(torch.float32), batch.y.to(torch.float32))
            losses += loss.item()
            y_true.append(batch.y.detach().cpu().numpy())
            y_pred.append(output.detach().cpu().numpy())

    return losses, np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0).reshape(-1)


def run(lr=5e-5, epochs=200, adj_loss_coef=5e-6, param_l2_coef=5e-6, batch_size=32,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    logging.basicConfig(
        filename='DeepAntibody_wild.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logging.getLogger().addHandler(console_handler)

    current_path = Path.cwd()
    path = current_path / 'Dataset' / 'wild'
    dataset = AffinityDataset(root=str(path))
    embedding = np.load(str(current_path / 'wild_embeddings' / 'ProtTrans_wild.npy'))

    spliter = KFold(n_splits=10, shuffle=True, random_state=42)
    dataset_size = len(dataset)

    fold_loss_train = {}
    fold_loss_val = {}

    for fold, (train_indices, val_indices) in enumerate(spliter.split(range(dataset_size))):
        logging.info("Fold {}, train size {}, test size {}".format(fold + 1, len(train_indices), len(val_indices)))
        fold_loss_train[str(fold + 1)] = []
        fold_loss_val[str(fold + 1)] = []

        model = MuLAAIP()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        loss_fn = nn.MSELoss()

        train_data = dataset[train_indices.tolist()]
        val_data = dataset[val_indices.tolist()]

        train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            train_loss, y_true, y_pred = train_epoch(model=model,
                                                     device=device,
                                                     dataloader=train_data_loader,
                                                     loss_fn=loss_fn,
                                                     optimizer=optimizer,
                                                     param_l2_coef=param_l2_coef,
                                                     embedding=embedding,
                                                     adj_loss_coef=adj_loss_coef)
            avg_train_loss = train_loss / len(train_data_loader)
            train_mae, train_corr, train_rmse = evaluate_metric(y_true, y_pred)

            val_loss, y_true, y_pred = valid_epoch(model=model,
                                                   device=device,
                                                   dataloader=val_data_loader,
                                                   loss_fn=loss_fn,
                                                   embedding=embedding)
            avg_val_loss = val_loss / len(val_data_loader)
            val_mae, val_corr, val_rmse = evaluate_metric(y_true, y_pred)


            logging.info(
                f"Epoch {epoch + 1}: Train MSE: {avg_train_loss:.4f}, "
                f"MAE: {train_mae:.4f}, "
                f"PCC: {train_corr:.4f}, "
                f"RMSE: {train_rmse:.4f}; "
                f"Val MSE: {avg_val_loss:.4f}, "
                f"MAE: {val_mae:.4f}, "
                f"PCC: {val_corr:.4f}, "
                f"RMSE: {val_rmse:.4f}"
            )


if __name__ == "__main__":
    run()

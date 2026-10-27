import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import anndata as ad

from .ST_utils import fix_seed
from .model import DecAlignSpatial,DecAlignSpatial_unpaired

import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F

def batch_balanced_mse(prediction,target,batch_codes,):
    """
    Calculate the MSE for each batch, then take the average across all batches.
    prediction: [N, D]
    target: [N, D]
    batch_codes: [N]
    """
    batch_losses = []

    for batch_id in torch.unique(batch_codes):
        mask = batch_codes == batch_id
        if mask.sum() == 0:
            continue

        loss_b = F.mse_loss(prediction[mask],target[mask],reduction="mean")
        batch_losses.append(loss_b)

    if len(batch_losses) == 0:
        return prediction.new_tensor(0.0)

    return torch.stack(batch_losses).mean()


def train(adata_omic1, adata_omic2, hidden_dims=16, output_dims=16, n_epochs=1000, lr=0.001,
                            weight_decay=1e-4,ot_num_iter=50, alpha_recon=1, beta_dec=0.1, gamma_homo=0.1,lambda_private=0.1, key_added='SIMPLEOmics',
                            num_prototype=5, num_head=4,random_seed=2026, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                            ):
    
    fix_seed(random_seed)
    if 'feat' in adata_omic1.obsm:
        x_omic1_np = adata_omic1.obsm['feat']
    else:
        x_omic1_np = adata_omic1.X
        
    if 'feat' in adata_omic2.obsm:
        x_omic2_np = adata_omic2.obsm['feat']
    else:
        x_omic2_np = adata_omic2.X
    
    x_omic1 = torch.FloatTensor(x_omic1_np.toarray() if sp.issparse(x_omic1_np) else x_omic1_np).to(device)
    x_omic2 = torch.FloatTensor(x_omic2_np.toarray() if sp.issparse(x_omic2_np) else x_omic2_np).to(device)

    edge_index_omic1 = adata_omic1.uns['edge_index']
    if isinstance(edge_index_omic1, np.ndarray):
        edge_index_omic1 = torch.tensor(edge_index_omic1, dtype=torch.long)
    elif isinstance(edge_index_omic1, (list, tuple)):
        edge_index_omic1 = torch.tensor(np.array(edge_index_omic1), dtype=torch.long)
    edge_index_omic1 = edge_index_omic1.to(device)
    edge_index_omic2 = adata_omic2.uns['edge_index']
    if isinstance(edge_index_omic2, np.ndarray):
        edge_index_omic2 = torch.tensor(edge_index_omic2, dtype=torch.long)
    elif isinstance(edge_index_omic2, (list, tuple)):
        edge_index_omic2 = torch.tensor(np.array(edge_index_omic2), dtype=torch.long)
    edge_index_omic2 = edge_index_omic2.to(device)

    
    if 'batch' in adata_omic1.obs:
        batch_codes_np = pd.Categorical(adata_omic1.obs["batch"]).codes
        batch_codes = torch.tensor(batch_codes_np,dtype=torch.long,device=device)
        assert np.array_equal(adata_omic1.obs["batch"].astype(str).to_numpy(),adata_omic2.obs["batch"].astype(str).to_numpy())

    class ArgsConfig:
        input_dim_omic1 = x_omic1.shape[1]
        input_dim_omic2 = x_omic2.shape[1]
        recon_dim_omic1 = x_omic1.shape[1]
        recon_dim_omic2 = x_omic2.shape[1]
        hidden_dim = hidden_dims
        output_dim = output_dims
        num_prototypes = num_prototype
        num_heads = num_head
        lambda_ot = 0.1
        ot_num_iters = ot_num_iter
        omic1_pos = adata_omic1.obsm['spatial']
        omic2_pos = adata_omic2.obsm['spatial']

    args = ArgsConfig()
    model = DecAlignSpatial(args).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print("training SIMPLEOmics...")
    for epoch in tqdm(range(n_epochs)):
        model.train()
        optimizer.zero_grad()
        res = model(x_omic1, x_omic2, edge_index_omic1, edge_index_omic2)
        
        dec_loss = res['dec_loss']
        homo_loss = res['homo_loss']
        recon_1 = res['recon_1']
        recon_2 = res['recon_2']
        
        recon_1_shared = res['recon_1_shared']
        recon_2_shared = res['recon_2_shared']
        recon_1_private = res['recon_1_private']
        recon_2_private = res['recon_2_private']
                
        homo_1 = res['homo_1']
        homo_2 = res['homo_2']

        # reconstruction loss and private loss
        loss_recon_1 = F.mse_loss(recon_1, x_omic1)
        loss_recon_2 = F.mse_loss(recon_2, x_omic2)
        residual_1 = x_omic1 - recon_1_shared.detach()
        residual_2 = x_omic2 - recon_2_shared.detach()

        loss_private_recon = F.mse_loss(recon_1_private, residual_1)+ F.mse_loss(recon_2_private, residual_2)
        loss_recon = loss_recon_1 + loss_recon_2

        if 'batch' in adata_omic1.obs:
            loss_recon_1 = batch_balanced_mse(recon_1,x_omic1,batch_codes)
            loss_recon_2 = batch_balanced_mse(recon_2,x_omic2,batch_codes)
            loss_recon = loss_recon_1 + loss_recon_2
            loss_private_recon_1 = batch_balanced_mse(recon_1_private,residual_1,batch_codes)
            loss_private_recon_2 = batch_balanced_mse(recon_2_private,residual_2,batch_codes)
            loss_private_recon = loss_private_recon_1 + loss_private_recon_2

        # total loss
        total_loss = alpha_recon * loss_recon +beta_dec * dec_loss + gamma_homo * homo_loss + lambda_private * loss_private_recon
        total_loss.backward()
        optimizer.step()

    # save results   
    model.eval()
    with torch.no_grad():
        res = model(x_omic1, x_omic2, edge_index_omic1, edge_index_omic2)
        final_embedding = res['final_rep']
        emb_homo = res['emb_homo']
        omic1_embedding = res['emb_omic1']
        omic2_embedding = res['emb_omic2']
        homo_1 = res['homo_1']
        homo_2 = res['homo_2']
        final_emb_np = final_embedding.cpu().numpy()
    adata_omic1.obsm[key_added] = final_emb_np
    adata_omic1.obsm['homo'] = emb_homo.cpu().numpy()
    adata_omic1.obsm['omic1_embedding'] = omic1_embedding.cpu().numpy()
    adata_omic1.obsm['omic2_embedding'] = omic2_embedding.cpu().numpy()
    adata_omic1.obsm['homo_1']=homo_1.cpu().numpy()
    adata_omic1.obsm['homo_2']=homo_2.cpu().numpy()
    return adata_omic1, adata_omic2


def train_unpaired(adata_omic1, adata_omic2, hidden_dims=16, output_dims=16, n_epochs=1000, lr=0.001,
                weight_decay=1e-4,ot_num_iter=50, alpha_recon=1, beta_dec=0.1, gamma_homo=0.1,lambda_private=0.1,
                num_prototype=5, num_head=4,random_seed=2026, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                ):
    
    fix_seed(random_seed)
    if 'feat' in adata_omic1.obsm:
        x_omic1_np = adata_omic1.obsm['feat']
    else:
        x_omic1_np = adata_omic1.X
        
    if 'feat' in adata_omic2.obsm:
        x_omic2_np = adata_omic2.obsm['feat']
    else:
        x_omic2_np = adata_omic2.X
    
    x_omic1 = torch.FloatTensor(x_omic1_np.toarray() if sp.issparse(x_omic1_np) else x_omic1_np).to(device)
    x_omic2 = torch.FloatTensor(x_omic2_np.toarray() if sp.issparse(x_omic2_np) else x_omic2_np).to(device)

    edge_index_omic1 = adata_omic1.uns['edge_index']
    if isinstance(edge_index_omic1, np.ndarray):
        edge_index_omic1 = torch.tensor(edge_index_omic1, dtype=torch.long)
    elif isinstance(edge_index_omic1, (list, tuple)):
        edge_index_omic1 = torch.tensor(np.array(edge_index_omic1), dtype=torch.long)
    edge_index_omic1 = edge_index_omic1.to(device)
    edge_index_omic2 = adata_omic2.uns['edge_index']
    if isinstance(edge_index_omic2, np.ndarray):
        edge_index_omic2 = torch.tensor(edge_index_omic2, dtype=torch.long)
    elif isinstance(edge_index_omic2, (list, tuple)):
        edge_index_omic2 = torch.tensor(np.array(edge_index_omic2), dtype=torch.long)
    edge_index_omic2 = edge_index_omic2.to(device)
    print(x_omic2.shape[1])
    print(x_omic1.shape[1])

    class ArgsConfig:
        input_dim_omic1 = x_omic1.shape[1]
        input_dim_omic2 = x_omic2.shape[1]
        recon_dim_omic1 = x_omic1.shape[1]
        recon_dim_omic2 = x_omic2.shape[1]
        hidden_dim = hidden_dims
        output_dim = output_dims
        num_prototypes = num_prototype
        num_heads = num_head
        lambda_ot = 0.1
        ot_num_iters = ot_num_iter
        omic1_pos = adata_omic1.obsm['spatial']
        omic2_pos = adata_omic2.obsm['spatial']

    args = ArgsConfig()
    model = DecAlignSpatial_unpaired(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # OT is not enabled for the first 50% of the epochs, and then gradually increased for the next 50% of the epochs.
    # This is controlled by the `warmup_epochs` and `ramp_epochs` variables.
    warmup_epochs = int(n_epochs * 0.5)
    ramp_epochs = max(1, int(n_epochs * 0.5))
    print("training SIMPLEOmics...")
    for epoch in tqdm(range(n_epochs)):
        model.train()
        optimizer.zero_grad()
        res = model(x_omic1, x_omic2, edge_index_omic1, edge_index_omic2)
        
        dec_loss = res['dec_loss']
        homo_loss = res['homo_loss']
        recon_1 = res['recon_1']
        recon_2 = res['recon_2']
        recon_1_shared = res['recon_1_shared']
        recon_2_shared = res['recon_2_shared']
        recon_1_private = res['recon_1_private']
        recon_2_private = res['recon_2_private']
        homo_1 = res['homo_1']
        homo_2 = res['homo_2']
        
        # reconstruction loss and private loss
        loss_recon_1 = F.mse_loss(recon_1, x_omic1)
        loss_recon_2 = F.mse_loss(recon_2, x_omic2)
        residual_1 = x_omic1 - recon_1_shared.detach()
        residual_2 = x_omic2 - recon_2_shared.detach()
        loss_private_recon = F.mse_loss(recon_1_private, residual_1)+ F.mse_loss(recon_2_private, residual_2)
        loss_recon = loss_recon_1 + loss_recon_2

        if epoch < warmup_epochs:
            ot_factor = 0.0
        else:
            ot_factor = min(1.0,(epoch - warmup_epochs) / ramp_epochs)

        effective_gamma = gamma_homo * ot_factor
        # total loss
        total_loss = alpha_recon * loss_recon + beta_dec * dec_loss + effective_gamma * homo_loss + lambda_private * loss_private_recon
        total_loss.backward()
        optimizer.step()

    # save results
    model.eval()
    with torch.no_grad():
        res = model(x_omic1, x_omic2, edge_index_omic1, edge_index_omic2)
        final_embedding_1 = res['final_rep_1']
        final_embedding_2 = res['final_rep_2']
        homo_1 = res['homo_1']
        homo_2 = res['homo_2']
        
    adata_omic1.obsm['omic1_embedding'] = final_embedding_1.cpu().numpy()
    adata_omic2.obsm['omic2_embedding'] = final_embedding_2.cpu().numpy()
    adata_omic1.obsm['homo_embedding'] = homo_1.cpu().numpy()
    adata_omic2.obsm['homo_embedding'] = homo_2.cpu().numpy()

    return adata_omic1, adata_omic2

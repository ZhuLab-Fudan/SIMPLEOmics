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
        recon_rna = res['recon_1']
        recon_adt = res['recon_2']
        
        recon_1_shared = res['recon_1_shared']
        recon_2_shared = res['recon_2_shared']
        recon_1_private = res['recon_1_private']
        recon_2_private = res['recon_2_private']
                
        homo_1 = res['homo_1']
        homo_2 = res['homo_2']

        # reconstruction loss and private loss
        loss_recon_rna = F.mse_loss(recon_rna, x_omic1)
        loss_recon_adt = F.mse_loss(recon_adt, x_omic2)
        rna_residual = x_omic1 - recon_1_shared.detach()
        adt_residual = x_omic2 - recon_2_shared.detach()

        loss_private_recon = F.mse_loss(recon_1_private, rna_residual)+ F.mse_loss(recon_2_private, adt_residual)
        loss_recon = loss_recon_rna + loss_recon_adt

        if 'batch' in adata_omic1.obs:
            loss_recon_rna = batch_balanced_mse(recon_rna,x_omic1,batch_codes)
            loss_recon_adt = batch_balanced_mse(recon_adt,x_omic2,batch_codes)
            loss_recon = loss_recon_rna + loss_recon_adt
            loss_private_recon_rna = batch_balanced_mse(recon_1_private,rna_residual,batch_codes)
            loss_private_recon_adt = batch_balanced_mse(recon_2_private,adt_residual,batch_codes)
            loss_private_recon = loss_private_recon_rna + loss_private_recon_adt

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

def integrate_homo_embeddings(emb1, emb2, obs_names1, obs_names2):
    """
    integrate two modalities' homo_embedding, support mosaic (non-aligned) integration.
    """

    if not isinstance(emb1, torch.Tensor):
        emb1 = torch.tensor(emb1, device=emb1.device if hasattr(emb1, 'device') else 'cpu')
    if not isinstance(emb2, torch.Tensor):
        emb2 = torch.tensor(emb2, device=emb2.device if hasattr(emb2, 'device') else 'cpu')
    
    obs1 = pd.Index(obs_names1)
    obs2 = pd.Index(obs_names2)
    common_names = sorted(obs1.intersection(obs2))
    map1 = {name: i for i, name in enumerate(obs_names1)}
    map2 = {name: i for i, name in enumerate(obs_names2)}
    final_homo1 = emb1.clone()
    final_homo2 = emb2.clone()
    
    if len(common_names) > 0:
        idx1_common = [map1[name] for name in common_names]
        idx2_common = [map2[name] for name in common_names]
        emb1_common = emb1[idx1_common]
        emb2_common = emb2[idx2_common]
        merged = ( emb1_common +  emb2_common) / 2
        final_homo1[idx1_common] = merged
        final_homo2[idx2_common] = merged

    return final_homo1, final_homo2


def build_joint_adata_for_unpaired(adata_omic1, adata_omic2, final_homo_1, final_homo_2,
                                    final_rep_1=None, final_rep_2=None, key_added="SIMPLEOmics"):


    if isinstance(final_homo_1, torch.Tensor):
        homo_1_np = final_homo_1.detach().cpu().numpy()
    else:
        homo_1_np = np.asarray(final_homo_1)

    if isinstance(final_homo_2, torch.Tensor):
        homo_2_np = final_homo_2.detach().cpu().numpy()
    else:
        homo_2_np = np.asarray(final_homo_2)
    obs_names_rna = pd.Index(adata_omic1.obs_names.astype(str))
    obs_names_adt = pd.Index(adata_omic2.obs_names.astype(str))
    assert obs_names_rna.is_unique, "RNA obs_names are not unique."
    assert obs_names_adt.is_unique, "ATAC obs_names are not unique."

    common_names = obs_names_rna.intersection(obs_names_adt)
    rna_only_names = obs_names_rna.difference(obs_names_adt)
    adt_only_names = obs_names_adt.difference(obs_names_rna)

    print("======================================")
    print("Building joint AnnData")
    print("RNA spots:", len(obs_names_rna))
    print("ATAC spots:", len(obs_names_adt))
    print("Common spots:", len(common_names))
    print("RNA-only spots:", len(rna_only_names))
    print("ATAC-only spots:", len(adt_only_names))
    print(
        "Final unique spots:",
        len(common_names) + len(rna_only_names) + len(adt_only_names)
    )
    print("======================================")

    # ============================================================
    # 3. 建立名称 -> index 的映射
    # ============================================================
    map_rna = {name: i for i, name in enumerate(obs_names_rna)}
    map_adt = {name: i for i, name in enumerate(obs_names_adt)}
    joint_names = list(obs_names_rna) + list(adt_only_names)

    obs_rna = adata_omic1.obs.copy()
    obs_adt_only = adata_omic2.obs.loc[adt_only_names].copy()

    obs_rna["original_obs_name"] = obs_rna.index.astype(str)
    obs_adt_only["original_obs_name"] = obs_adt_only.index.astype(str)
    obs_rna["modality"] = "RNA"

    if len(common_names) > 0:
        obs_rna.loc[common_names, "modality"] = "paired"

    obs_adt_only["modality"] = "ATAC"

    # concat
    obs_joint = pd.concat([obs_rna,obs_adt_only],axis=0)
    obs_joint = obs_joint.loc[joint_names].copy()

    joint_homo_rna = homo_1_np
    adt_only_idx = [map_adt[name] for name in adt_only_names]

    if len(adt_only_idx) > 0:
        joint_homo_adt_only = homo_2_np[adt_only_idx]
        joint_homo = np.concatenate([joint_homo_rna,joint_homo_adt_only],axis=0)
    else:
        joint_homo = joint_homo_rna.copy()

    # ============================================================
    # 7. 创建 AnnData
    # ============================================================
    adata_joint = ad.AnnData(X=np.zeros((len(joint_names), 1),dtype=np.float32),obs=obs_joint)

    # final shared representation
    adata_joint.obsm["final_homo"] = joint_homo
    adata_joint.obsm[key_added] = joint_homo

    if final_rep_1 is not None and final_rep_2 is not None:

        if isinstance(final_rep_1, torch.Tensor):
            rep1_np = final_rep_1.detach().cpu().numpy()
        else:
            rep1_np = np.asarray(final_rep_1)

        if isinstance(final_rep_2, torch.Tensor):
            rep2_np = final_rep_2.detach().cpu().numpy()
        else:
            rep2_np = np.asarray(final_rep_2)

        if rep1_np.shape[1] == rep2_np.shape[1]:

            joint_rep = []

            for name in joint_names:

                in_rna = name in map_rna
                in_adt = name in map_adt

                if in_rna and in_adt:
                    # paired spot
                    z = (
                        rep1_np[map_rna[name]] +
                        rep2_np[map_adt[name]]
                    ) / 2.0

                elif in_rna:
                    z = rep1_np[map_rna[name]]

                else:
                    z = rep2_np[map_adt[name]]

                joint_rep.append(z)

            adata_joint.obsm["final_rep"] = np.stack(
                joint_rep,
                axis=0
            )

        else:
            print(
                "Warning: final_rep_1 and final_rep_2 have "
                "different dimensions. Skip building joint final_rep."
            )

    # ============================================================
    # 9. spatial coordinates
    #
    # RNA rows：
    # common / RNA-only -> RNA coordinates
    #
    # ATAC-only：
    # -> ATAC coordinates
    # ============================================================
    if (
        "spatial" in adata_omic1.obsm
        and "spatial" in adata_omic2.obsm
    ):

        spatial_rna = np.asarray(
            adata_omic1.obsm["spatial"]
        )

        spatial_adt = np.asarray(
            adata_omic2.obsm["spatial"]
        )

        if spatial_rna.shape[1] == spatial_adt.shape[1]:

            spatial_adt_only = spatial_adt[
                adt_only_idx
            ]

            adata_joint.obsm["spatial"] = np.concatenate(
                [
                    spatial_rna,
                    spatial_adt_only
                ],
                axis=0
            )

    # ============================================================
    # 10. final checks
    # ============================================================
    assert adata_joint.n_obs == joint_homo.shape[0]

    assert list(
        adata_joint.obs["original_obs_name"]
    ) == joint_names

    print(
        "Final adata_joint shape:",
        adata_joint.shape
    )

    return adata_joint

def train_unpaired(adata_omic1, adata_omic2, hidden_dims=16, output_dims=16, n_epochs=1000, lr=0.001,
                                    key_added='SIMPLEOmics', weight_decay=1e-4,ot_num_iter=50, alpha_recon=1, beta_dec=0.1, gamma_homo=0.1,lambda_private=0.1,
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
        recon_rna = res['recon_1']
        recon_adt = res['recon_2']
        recon_1_shared = res['recon_1_shared']
        recon_2_shared = res['recon_2_shared']
        recon_1_private = res['recon_1_private']
        recon_2_private = res['recon_2_private']
        homo_1 = res['homo_1']
        homo_2 = res['homo_2']
        
        # reconstruction loss and private loss
        loss_recon_rna = F.mse_loss(recon_rna, x_omic1)
        loss_recon_adt = F.mse_loss(recon_adt, x_omic2)
        rna_residual = x_omic1 - recon_1_shared.detach()
        adt_residual = x_omic2 - recon_2_shared.detach()
        loss_private_recon = F.mse_loss(recon_1_private, rna_residual)+ F.mse_loss(recon_2_private, adt_residual)
        loss_recon = loss_recon_rna + loss_recon_adt

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
        final_homo_1,final_homo_2 = integrate_homo_embeddings(homo_1, homo_2, adata_omic1.obs_names, adata_omic2.obs_names)
        adata_joint = build_joint_adata_for_unpaired(adata_omic1,adata_omic2,final_homo_1,final_homo_2,final_rep_1=final_embedding_1,final_rep_2=final_embedding_2,key_added=key_added)
        
    adata_omic1.obsm['omic1_embedding'] = final_embedding_1.cpu().numpy()
    adata_omic2.obsm['omic2_embedding'] = final_embedding_2.cpu().numpy()
    adata_omic1.obsm['homo_embedding'] = homo_1.cpu().numpy()
    adata_omic2.obsm['homo_embedding'] = homo_2.cpu().numpy()

    return adata_omic1, adata_omic2,adata_joint
import os
import random
import pandas as pd
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors

import scipy
import scipy.sparse as sp
import anndata
from harmony import harmonize
from muon import atac as ac

from typing import Optional, Tuple
import torch
from torch.backends import cudnn

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    torch.use_deterministic_algorithms(True)


def harmony(adata, feature_labels, batch_labels, use_gpu=True):
    """
    Perform batch correction using harmony-pytorch.
    """

    df_batches = pd.DataFrame(np.reshape(adata.obs[batch_labels].values, (-1, 1)), columns=[batch_labels])
    bc_latent = harmonize(adata.obsm[feature_labels], df_batches, batch_key=batch_labels,
                        use_gpu=use_gpu, verbose=True, max_iter_harmony=20, theta=10)
    adata.obsm[feature_labels] = bc_latent
    

def clr_normalize_each_cell(adata, inplace=True):
    
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata 

def pca(adata,use_reps=None,n_comps=10,random_state=2026,save_model=True,model_key="pca_model"):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix

    pca = PCA(n_components=n_comps,random_state=random_state)

    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)

    if save_model:
       adata.uns[model_key] = pca
    
    return feat_pca


def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf   

def Cal_Net(adata, k=6, spatial_metric='euclidean', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    k
        The number of nearest neighbors when model='KNN'
    
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    
    if verbose: print(f'------Calculating Spatial Graph (k={k}, metric={spatial_metric})...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    nbrs_s = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(coor)
    distances_s, indices_s = nbrs_s.kneighbors(coor)
    s_cell1 = np.repeat(np.arange(adata.n_obs), k)
    s_cell2 = indices_s[:, 1:].flatten()
    adata.uns['edge_index'] = np.vstack((s_cell1, s_cell2))
    adj_spatial = sp.coo_matrix((np.ones(len(s_cell1)), (s_cell1, s_cell2)), shape=(adata.n_obs, adata.n_obs))
    adata.uns['adj'] = adj_spatial
    if verbose:
        print(f'Spatial edges: {adata.uns["edge_index"].shape[1]}')
    return adata


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='SIMPLEOmics', random_seed=2026):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(adata.obsm[used_obsm], num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    adata.obs['mclust'] = mclust_res.astype(int).astype(str)
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata
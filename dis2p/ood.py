from typing import List, Tuple
import numpy as np
import torch
from sklearn.metrics import r2_score
from anndata import AnnData
import scanpy as sc

from .dis2pvi import Dis2pVI
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def ood_for_given_covs(
        adata: AnnData,
        cats: List[str],
        vi_cls=Dis2pVI,
        model_name='',
        pre_path: str = '.',
        cov_idx: int = 0,
        cov_value: str = '',
        cov_value_cf: str = '',
        other_covs_values: Tuple = (0,),
        remove_all_samples_with_other_covs_values: bool = True,
        **train_dict,
):

    cov_name = cats[cov_idx]
    other_cats = [c for c in cats if c != cov_name]

    train_sub_idx = []
    true_idx = []
    source_sub_idx = []

    if remove_all_samples_with_other_covs_values:
        # remove all cells with other_covs_values

        for i in range(adata.n_obs):
            cov_value_i = adata[i].obs[cats[cov_idx]][0]

            if all(adata[i].obs[c][0] == other_covs_values[k] for k, c in enumerate(other_cats)):
                if cov_value_i == cov_value_cf:
                    true_idx.append(i)
                elif cov_value_i == cov_value:
                    source_sub_idx.append(i)
            else:
                train_sub_idx.append(i)

    else:
        # remove only cells with other_covs_values and cov_value_cf

        for i in range(adata.n_obs):
            cov_value_i = adata[i].obs[cats[cov_idx]][0]

            if all(adata[i].obs[c][0] == other_covs_values[k] for k, c in enumerate(other_cats)):
                if cov_value_i == cov_value_cf:
                    true_idx.append(i)
                else:
                    train_sub_idx.append(i)
                    if cov_value_i == cov_value:
                        source_sub_idx.append(i)
            else:
                train_sub_idx.append(i)

    adata_sub = adata[train_sub_idx]

    cov_str_cf = cats[cov_idx] + ' = ' + str(cov_value) + ' to ' + str(cov_value_cf)
    other_covs_str = ', '.join(c + ' = ' + str(other_covs_values[k]) for k, c in enumerate(other_cats))

    try:
        model = vi_cls.load(f"{pre_path}/{model_name}", adata=adata_sub)
    except:
        adata_sub = adata_sub.copy()
        vi_cls.setup_anndata(
            adata_sub,
            layer='counts',
            categorical_covariate_keys=cats,
            continuous_covariate_keys=[]
        )
        model = vi_cls(adata_sub)
        model.train(**train_dict)
        model.save(f"{pre_path}/{model_name}")

    source_adata = adata[source_sub_idx]

    px_cf_mean_pred, px_cf_variance_pred = model.predict_given_covs(adata=source_adata, cats=cats, cov_idx=cov_idx,
                                                                    cov_value_cf=cov_value_cf)

    px_cf_mean_pred, px_cf_variance_pred = px_cf_mean_pred.to('cpu'), px_cf_variance_pred.to('cpu')

    true_x_count = torch.tensor(adata.layers["counts"][true_idx].toarray())
    true_x_counts_mean = torch.mean(true_x_count, dim=0)

    true_x_counts_variance = torch.sub(true_x_count, true_x_counts_mean)
    true_x_counts_variance = torch.pow(true_x_counts_variance, 2)
    true_x_counts_variance = torch.mean(true_x_counts_variance, dim=0)

    true_x_counts_mean, true_x_counts_variance = true_x_counts_mean.to('cpu'), true_x_counts_variance.to('cpu')

    print(f'Counterfactual prediction for {cov_str_cf}, and {other_covs_str}')

    return true_x_counts_mean, true_x_counts_variance, px_cf_mean_pred, px_cf_variance_pred


def r2_eval(adata, cov_name, cov_value_cf, true_x_counts_stat, px_cf_stat_pred, n_top_deg: int = 20):
    adata.var['name'] = adata.var.index
    sc.tl.rank_genes_groups(adata, cov_name, method='wilcoxon', key_added="wilcoxon")
    ranked_genes = sc.get.rank_genes_groups_df(adata, group=cov_value_cf, key='wilcoxon', gene_symbols='name')
    ranked_genes_names = ranked_genes[ranked_genes['name'].notnull()]['name']
    deg_names = ranked_genes_names[:n_top_deg]
    deg_idx = [i for i, _ in enumerate(adata.var['name']) if adata.var['name'][i] in list(deg_names)]

    r2 = r2_score(true_x_counts_stat, px_cf_stat_pred)
    r2_deg = r2_score(true_x_counts_stat[deg_idx], px_cf_stat_pred[deg_idx])

    try:
        r2_log = r2_score(np.log1p(true_x_counts_stat), np.log1p(px_cf_stat_pred))
        r2_log_deg = r2_score(np.log1p(true_x_counts_stat[deg_idx]), np.log1p(px_cf_stat_pred[deg_idx]))
    except:
        r2_log = r2_score(torch.log1p_(true_x_counts_stat), torch.log1p_(px_cf_stat_pred))
        r2_log_deg = r2_score(torch.log1p_(true_x_counts_stat[deg_idx]), torch.log1p_(px_cf_stat_pred[deg_idx]))

    print('All Genes')
    print(f'R2 = {r2:.4f}')
    print(f'R2 log = {r2_log:.4f}')

    print(f'DE Genes (n_top={n_top_deg})')
    print(f'R2 = {r2_deg:.4f}')
    print(f'R2 log = {r2_log_deg:.4f}')

    return r2, r2_log, r2_deg, r2_log_deg

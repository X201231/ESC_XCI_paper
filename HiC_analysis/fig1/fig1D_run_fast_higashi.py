import pandas as pd
import numpy as np
import pickle

from umap import UMAP
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import manifold
import sys
sys.path.append('/share/home/hxie/miniconda3/envs/hic/lib/python3.10/site-packages')

metadata = pd.read_csv("/share/Data/hxie/project/202209/esc_xwliu/esc1014_halfday/RNA_analysis/s1014_metadata_X_noM_noXistneg.csv")

label_info = {
    'cellcycle_threshold': metadata["cellcycle_threshold"].values.tolist(),
    'day': metadata["day"].values.tolist(),
 #   'batch_id': metadata["cellcycle_threshold"].values.tolist(),
}

cellnames = metadata["cellname"].values
filelists = ["/share/Data/hxie/project/202209/esc_xwliu/esc1014_halfday/HiC_analysis/visual/scHiC_analysis/processed/dip_pairs_all/" + cellname + ".dip.pairs.gz" for cellname in cellnames]

from fasthigashi.FastHigashi_Wrapper import *

config_path = "./HiRES_XCI/config.JSON"
# Initialize the model
model = FastHigashi(config_path=config_path,
	             path2input_cache=None,
	             path2result_dir=None,
	             off_diag=100,
	             filter=True,
	             do_conv=False,
	             do_rwr=False,
	             do_col=False,
	             no_col=False)

model.fast_process_data()

model.prep_dataset(batch_norm=False)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

model.run_model(dim1=0.6,
                rank=8,
                n_iter_parafac=1,
                extra="")

embedding = model.fetch_cell_embedding(final_dim=8,restore_order=True)
embedding = model.correct_batch_linear(var_to_regress_name='cellcycle_threshold')

spec_embedding = manifold.SpectralEmbedding(n_components=2,n_neighbors=30).fit_transform(embedding["embed_l2_norm_correct_cellcycle_threshold"])

# df = pd.DataFrame(embedding["embed_l2_norm_correct_cellcycle_threshold"])
df = pd.DataFrame(spec_embedding)
df.to_csv("./embedding_dim2.csv", index=True)
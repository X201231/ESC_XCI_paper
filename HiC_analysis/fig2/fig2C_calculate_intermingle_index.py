import sys
import numpy as np
import pandas as pd

sys.path.append('/shareb/zliu/analysis/')
sys.path.append('/shareb/zliu/analysis/CHARMtools')
from CHARMtools import Cell3Ddev as Cell3D
# https://github.com/skelviper/CHARMtools

import imp
imp.reload(Cell3D)
import os

from multiprocessing import Pool

def process_cell(cell):
    cell_path = f"/share/Data/hxie/project/202209/esc_xwliu/esc0312_tidyup/structure/reRun_3D/processed/{cell}/3d_info/clean.200k.0.3dg"
    cell_instance = Cell3D.Cell3D(tdg_path=cell_path, resolution=200000, cellname=cell)
    cell_instance.calc_intermingling()
    output_path = f'/share/Data/hxie/project/202209/esc_xwliu/esc0615_version2/HiC_analysis/intermingle/data/200k/{cell}_200k.h5'
    cell_instance.tdg.to_hdf(output_path, key='df', mode='w', format='table')

if __name__ == '__main__':
    metadata = pd.read_csv("/share/Data/hxie/project/202209/esc_xwliu/esc1014_halfday/RNA_analysis/s1014_metadata_X_noM_noXistneg.csv")
    metadata = metadata.query('cellname != "d4A8"')
    rmsd = pd.read_csv("/share/Data/hxie/project/202209/esc_xwliu/esc0312_tidyup/structure/reRun_3D/stat/rmsd.csv", index_col = 0)
    rmsd = rmsd.query('`20k` <= 1.5')

    cellnames = metadata['cellname'].values
    cellnames = [x for x in cellnames if x in rmsd['cellname'].values]

    num_cores = os.cpu_count()

    with Pool(processes=num_cores) as pool:
        pool.map(process_cell, cellnames)
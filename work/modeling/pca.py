import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
import pandas as pd
from sklearn.preprocessing import normalize

from logger_settings import configure_logger
import logging
import os

configure_logger()
logger=logging.getLogger(__name__)

def load_npy_data(src_path):
    df = None
    name_list=[]
    for i,file_entry in enumerate(os.scandir(src_path)):
        if file_entry.name.endswith('.npy'):
            file = normalize(np.load(file_entry.path)).flatten()
            name = file_entry.name.rsplit('.', 1)[0]
            name_list.append(name)
            if df is None:
                df = pd.DataFrame(file)
            else:
                df[i]=file

    df = df.T
    df.columns = df.columns.astype(str)
    df.insert(0,"file_name",name_list)

    return df


"""def load_npy_data(src_path):
    df = None
    for i,file_entry in enumerate(os.scandir(src_path)):
        if file_entry.name.endswith('.npy'):
            file = normalize(np.load(file_entry.path)).flatten()
            name = file_entry.name.rsplit('.', 1)[0]
            file_as_list = list(file)
            row = [name] + file_as_list
            if df is None:
                df = pd.DataFrame(row)
            else:
                df[i]=row

    df = df.T
    df.columns = df.columns.astype(str)
    df.rename(columns={df.columns[0]: "file_name"})# doesnt work!!! # TODO fix df name

    return df"""


def main():
    src_path= r'D:\Clarifruit\cherry_stem\data\unet_data\training\2019-09-30_07-19-46\thres_0.4\bgr_histograms'
    data_frame= load_npy_data(src_path)
    print(data_frame.head())

def performe_pca(df, n_comp=20):

    logger.info('Running PCA ...')
    pca = IncrementalPCA(n_components=n_comp)
    X_pca = pca.fit_transform(df)
    logger.info('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())

    logger.info('Individual variance contributions:')
    for j in range(n_comp):
        logger.info(pca.explained_variance_ratio_[j])


if __name__ == '__main__':
    main()
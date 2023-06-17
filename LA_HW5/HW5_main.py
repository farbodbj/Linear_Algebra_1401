import pandas as pd
import numpy as np
from typing import Tuple
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATASET_PATH = "/media/farbod/2A8AD6C38AD68B29/Farbod/university/term 4/Linear algebra/Homeworks/HW5/LA_HW5/nndb_flat.csv"
K_VALUES = [3, 5, 8, 10, 15]
SCATTER_MARKER_SIZE = 4

# You can check value returned by this function for
# your provided DataFrame "scaled_df", and "k" as
# number of used singular values
def get_percentage(scaled_df: pd.DataFrame, k: int):
    solver = PCA()
    solver.fit_transform(scaled_df)
    return solver.explained_variance_ratio_[:k].sum() * 100


def visualize(original_data, rcnst_list):
    # Create a figure with one row and len(rcnst_list) columns
    fig, axs = plt.subplots(1, len(rcnst_list), figsize=(15, 5))
    
    # Plot each set of reconstructed data in a separate subplot
    for i in range(len(rcnst_list)):
        axs[i].scatter(original_data[:, 0], original_data[:, 1], label='Original Data', s = SCATTER_MARKER_SIZE)
        axs[i].scatter(rcnst_list[i][:, 0], rcnst_list[i][:, 1], label=f"Reconstructed Data {i}", s = SCATTER_MARKER_SIZE)
        axs[i].set_title(f"Reconstructed Data {i}")
    
    # Set the axis labels and the legend for each subplot
    for ax in axs.flat:
        ax.set(xlabel='Dimension 1', ylabel='Dimension 2')
        ax.legend()

    # Adjust the spacing between subplots and display the figure
    plt.tight_layout()
    plt.show()
    


## Your Code Here!
## dropping columns which have no numericaly valuable information
df = pd.read_csv(DATASET_PATH)
df = df.drop(labels=
        ['FoodGroup',
        'ShortDescrip',
        'Descrip',
        'CommonName',
        'MfgName',
        'ScientificName'],
        axis=1
    )


# Needs no change
# scales and normalizes the data
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)


## Your Code Here!
# You should call your SVD function on scaled_df
# and also use the result for reconstructing the main matrix, for k = 3, 5, 8, 10, 15

def get_top(u:np.ndarray, s:np.ndarray, v:np.ndarray, k: int)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_top = u[:, :k]
    s_top = s[:k, :k]
    v_top = v[:k, :] #is this correct?
    
    return u_top, s_top, v_top
    
    
def reconstruct(u:np.ndarray, s:np.ndarray, v:np.ndarray)->np.ndarray:
    return u @ s @ v

# singular values are represented as an array and should be converted to a square matrix
u, s_arr, v = np.linalg.svd(scaled_df) 
s = np.pad(
    np.diag(s_arr),
    (
        (0, u.shape[0] - s_arr.shape[0]),
        (0, v.shape[1] - s_arr.shape[0])
    )
)



## Your Code Here!
# Print retained_percentage for values of k, and check with get_percentage()
reconstructed_list = []

for k in K_VALUES:
    recon = reconstruct(*get_top(u, s, v, k))
    print(f"Retained percentage for k={k} is {get_percentage(scaled_df, k)}")
    reconstructed_list.append(recon)


## Visualization
# give a list of your reconstructed data as well as original data to visualize()
visualize(scaled_df, reconstructed_list)

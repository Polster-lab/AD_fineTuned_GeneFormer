import scanpy as sc
import pandas as pd

adata = sc.read('PFC427_train_clean_anno.h5ad')

# print columns containing NaN values and how many there are
nan_obs = adata.obs.isna()
print("NaN values in .obs DataFrame:")
for column in nan_obs.columns:
    num_nan = nan_obs[column].sum()
    if num_nan > 0:
        print(f"{column}: {num_nan}")

print("\nFirst 5 elements of .obs:")
print(adata.obs.head())

nan_X = pd.DataFrame(adata.X).isna()
print("\nNaN values in .X array:")
for column in nan_X.columns:
    num_nan = nan_X[column].sum()
    if num_nan > 0:
        print(f"{column}: {num_nan}")

print("\nFirst 5 elements of .X:")
print(pd.DataFrame(adata.X).head())

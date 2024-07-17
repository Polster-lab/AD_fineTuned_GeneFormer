# %%
import datetime
import os
import scanpy as sc
from sklearn.model_selection import train_test_split
#from geneformer import Classifier
from custom_geneformer import CustomClassifier
from geneformer.classifier_utils import downsample_and_shuffle
from geneformer.perturber_utils import load_and_filter, downsample_and_sort
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

# %%
df = pd.read_csv('PFC427_train_clean_anno.csv')
df_resampled = pd.read_csv('PFC427_train_clean_anno_sample.csv')

# %%
le = LabelEncoder()
df['Major_Cell_Type_encoded'] = le.fit_transform(df['Major_Cell_Type'])

X = df.drop(columns=['Major_Cell_Type_encoded'])
y = df['Major_Cell_Type_encoded']

class_counts = y.value_counts()
ranks = class_counts.rank(method='min')

# define undersample factors
def get_undersample_factor(rank, max_rank):
    if rank <= 2:
        return 1.0  # no undersampling for the small clases
    else:
        # linear interpolation between 0.5 and 0.25
        factor = 0.22 + 0.78 * (max_rank - rank) / (max_rank - 1)
        return factor

max_rank = ranks.max()
sampling_strategy = {cls: int(count * get_undersample_factor(rank, max_rank)) for cls, rank, count in zip(class_counts.index, ranks, class_counts)}

resampler = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_resampled, y_resampled = resampler.fit_resample(X, y)

df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['Major_Cell_Type_encoded'])], axis=1)
print(df_resampled['Major_Cell_Type_encoded'].value_counts())

# %%
counts_df1 = df['Major_Cell_Type'].value_counts()
counts_df_resampled1 = df_resampled['Major_Cell_Type'].value_counts()

counts_df2 = df['AD_diagnosis'].value_counts()
counts_df_resampled2 = df_resampled['AD_diagnosis'].value_counts()

counts1 = pd.DataFrame({'Original': counts_df1, 'Sampled': counts_df_resampled1}).sort_values(by='Original', ascending=False)
counts2 = pd.DataFrame({'Original': counts_df2, 'Sampled': counts_df_resampled2}).sort_values(by='Original', ascending=False)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

counts1.plot(kind='bar', ax=axes[0])
axes[0].set_title('Major_Cell_Type')
counts2.plot(kind='bar', ax=axes[1])
axes[1].set_title('AD_diagnosis')

plt.tight_layout()
plt.show()

# %%
print("Original counts for 'AD_diagnosis':")
print(counts_df2)

print("\nResampled counts for 'AD_diagnosis':")
print(counts_df_resampled2)
import scanpy as sc

def save_obs_to_csv(h5ad_file, csv_file):
    sc.read_h5ad(h5ad_file).obs.to_csv(csv_file)

if __name__ == "__main__":

    save_obs_to_csv('PFC427_test_clean_anno.h5ad', 'PFC427_test_clean_anno_obs.csv')

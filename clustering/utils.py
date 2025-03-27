import pandas as pd
import numpy as np

def wide_to_long(df_wide):
    df_long = df_wide.copy()
    df_long['subject'] = df_long.index  # or use a real subject column if you have one
    df_long = df_long.melt(id_vars='subject', var_name='time', value_name='value')
    df_long['time'] = df_long['time'].astype(int)
    return df_long

def filter_small_clusters(df, cluster_column, min_member_count):
    labels = np.unique(df[cluster_column])
    print(f"found {labels} unique labels")
    members = [np.sum(df[cluster_column]==l) for l in labels]
    valid_labels = [l for l, m in zip(labels, members) if m >= min_member_count]
    df_filtered = df[df[cluster_column].isin(valid_labels)].copy()
    invalid_labels = [l for l in labels if l not in valid_labels]
    print(f"left with {len(valid_labels)} clusters")
    print(f"removed clusters {len(invalid_labels)} for not meeting the minimum member requirement {min_member_count}, totalling {np.sum([np.sum(df[cluster_column]==l) for l in invalid_labels])} metastsases")
    print(f"kept {valid_labels}, removed {invalid_labels}")
    return df_filtered
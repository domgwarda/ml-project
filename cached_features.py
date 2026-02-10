import pandas as pd
import os
from rapidfuzz import process, fuzz
from features import domains

CACHE_FILE = 'OtherData/similarity_cache.csv'

def join_domains(domains_set):
    return "-".join(sorted(list(domains_set)))

def load_popular():
    popular = pd.read_csv('OtherData/cloudflare-radar_top-1000-domains_20260119-20260126.csv')
    popular["domains"] = domains(popular["domain"])
    popular["joined_domains"] = popular["domains"].apply(join_domains)
    return popular["joined_domains"].tolist()

def add_cached_features(data):

    if os.path.exists(CACHE_FILE):
        cache = pd.read_csv(CACHE_FILE)
        data = data.merge(cache, on='URLs', how='left')
    else:
        data['best_match'] = None
        data['best_ratio'] = None

    nans = data['best_ratio'].isna()



    if nans.any():

        popular_list = load_popular()
        
        data.loc[nans, "_temp_joined"] = domains(data.loc[nans, "URLs"]).apply(join_domains)

        def compute_row(joined_domains):
            res = process.extractOne(joined_domains, popular_list, scorer=fuzz.ratio)
            return (res[0], res[1]) if res else ("", 0)

        results = data.loc[nans, '_temp_joined'].apply(compute_row)

        data.loc[nans, 'best_match'] = results.apply(lambda x: x[0])
        data.loc[nans, 'best_ratio'] = results.apply(lambda x: x[1])

        new_cache_entries = data.loc[nans, ['URLs', 'best_match', 'best_ratio']]
        if os.path.exists(CACHE_FILE):
            full_cache = pd.concat([pd.read_csv(CACHE_FILE), new_cache_entries])
        else:
            full_cache = new_cache_entries

        full_cache.drop_duplicates(subset=['URLs']).to_csv(CACHE_FILE, index=False)

    to_drop = ["_temp_joined"]
    data = data.drop(columns=[c for c in to_drop if c in data.columns], errors='ignore')
    return data
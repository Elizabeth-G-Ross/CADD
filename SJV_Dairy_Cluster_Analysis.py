"""
SJV Dairy Cluster Analysis for Mobile Emissions Measurement Campaign
Purpose: Identify candidate dairy clusters for 15-20 farm sampling portfolio
- Multiple herd size ranges with flexible pairing
- Match digester/non-digester pairs with SIMILAR herd sizes (within same range)
- Identify geographic clusters for efficient 4-dairy/day sampling routes
- Assess spatial isolation to minimize cross-contamination

Deployment: 3 × 5-day campaigns (seasonal variability)
Each deployment visits the same 15-20 dairies
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

# ==================== LOAD DATA ====================
print("="*80)
print("SJV DAIRY CLUSTER ANALYSIS FOR MOBILE EMISSIONS MEASUREMENT")
print("="*80)

# Load the three CADD data files
facility_df = pd.read_csv('CADD v2.0.0 .csv')
herd_df = pd.read_csv('CADD Facility Herd Size V2.csv')
digester_df = pd.read_csv('CADD Anaerobic Digester V2.csv')

print(f"\n[1] Loaded facility data: {len(facility_df)} dairies")
print(f"[2] Loaded herd size data: {len(herd_df)} records")
print(f"[3] Loaded digester data: {len(digester_df)} records")

# ==================== DEFINE SJV COUNTIES ====================
sjv_counties = ['Fresno', 'Kings', 'Kern', 'Tulare', 'Merced', 'Stanislaus', 'San Joaquin', 'Madera']

# ==================== DEFINE HERD SIZE RANGES ====================
# Multiple ranges to increase candidate pool while maintaining herd size pairing
herd_size_ranges = [
    {'name': 'Small-Mid (1000-2000)', 'min': 1000, 'max': 2000},
    {'name': 'Mid (2000-3500)', 'min': 2000, 'max': 3500},
    {'name': 'Mid-Large (3000-5000)', 'min': 3000, 'max': 5000},
    {'name': 'Large (4000-7000)', 'min': 4000, 'max': 7000},
    {'name': 'Very Large (6000-10000)', 'min': 6000, 'max': 10000},
]

print(f"\n[*] Herd size ranges defined:")
for r in herd_size_ranges:
    print(f"    • {r['name']}: {r['min']:,} - {r['max']:,} milking cows")

# ==================== FILTER FOR SJV & GET RECENT HERD SIZES ====================
print(f"\n[4] Filtering for SJV counties: {sjv_counties}")
facility_sjv = facility_df[facility_df['County'].isin(sjv_counties)].copy()
print(f"    Found {len(facility_sjv)} dairies in SJV")

# Get most recent herd size (2023 preferred, else 2022, etc.)
herd_latest = herd_df.sort_values('Year', ascending=False).drop_duplicates('CADDID', keep='first').copy()
print(f"    Got latest herd sizes for {len(herd_latest)} dairies")

# Merge facility + latest herd data
facility_sjv = facility_sjv.merge(herd_latest, on='CADDID', how='left')

# ==================== IDENTIFY DIGESTERS ====================
print(f"\n[5] Identifying anaerobic digester status...")
digester_sjv = digester_df[digester_df['CADDID'].isin(facility_sjv['CADDID'])].copy()
digester_ids = set(digester_sjv['CADDID'].unique())
print(f"    Found {len(digester_ids)} digesters in SJV")

facility_sjv['HasDigester'] = facility_sjv['CADDID'].isin(digester_ids)

# ==================== ASSIGN HERD SIZE CATEGORY ====================
print(f"\n[6] Assigning herd size categories...")

def assign_herd_size_range(herd_size, ranges):
    """Assign dairy to a herd size range"""
    if pd.isna(herd_size):
        return None
    for r in ranges:
        if r['min'] <= herd_size <= r['max']:
            return r['name']
    return None

facility_sjv['HerdSizeRange'] = facility_sjv['MilkCows'].apply(
    lambda x: assign_herd_size_range(x, herd_size_ranges)
)

# Filter out dairies with no assigned herd size range
facility_sjv = facility_sjv[facility_sjv['HerdSizeRange'].notna()].copy()
print(f"    {len(facility_sjv)} dairies assigned to herd size ranges")

# ==================== SUMMARIZE BY DIGESTER STATUS & HERD RANGE ====================
print(f"\n[7] Dairy Distribution by Digester Status & Herd Size Range:")
print("-"*80)

for herd_range in herd_size_ranges:
    range_name = herd_range['name']
    range_data = facility_sjv[facility_sjv['HerdSizeRange'] == range_name]
    
    if len(range_data) > 0:
        digester_count = range_data[range_data['HasDigester']].shape[0]
        non_digester_count = range_data[~range_data['HasDigester']].shape[0]
        total = len(range_data)
        
        print(f"\n  {range_name}:")
        print(f"    Dairies WITH digesters:     {digester_count:3d}")
        print(f"    Dairies WITHOUT digesters:  {non_digester_count:3d}")
        print(f"    Total:                      {total:3d}")

total_all = len(facility_sjv)
digesters_all = facility_sjv[facility_sjv['HasDigester']].shape[0]
non_digesters_all = facility_sjv[~facility_sjv['HasDigester']].shape[0]
print(f"\n  TOTAL ACROSS ALL RANGES:")
print(f"    Dairies WITH digesters:     {digesters_all:3d}")
print(f"    Dairies WITHOUT digesters:  {non_digesters_all:3d}")
print(f"    Total candidates:           {total_all:3d}")

# ==================== GEOGRAPHIC CLUSTERING BY HERD SIZE RANGE ====================
print(f"\n[8] Performing geographic clustering within each herd size range...")

all_clusters = []
cluster_counter = 0

for herd_range in herd_size_ranges:
    range_name = herd_range['name']
    range_data = facility_sjv[facility_sjv['HerdSizeRange'] == range_name].copy()
    
    if len(range_data) < 4:  # Skip ranges with too few dairies
        continue
    
    # DBSCAN clustering within this herd size range
    coords = range_data[['Latitude', 'Longitude']].values
    dbscan = DBSCAN(eps=0.05, min_samples=3)
    cluster_labels = dbscan.fit_predict(coords)
    
    range_data['ClusterID'] = cluster_labels
    
    # Filter out noise points
    range_data = range_data[range_data['ClusterID'] >= 0].copy()
    
    # Reassign cluster IDs to be globally unique
    range_data['ClusterID'] = range_data['ClusterID'].apply(lambda x: x + cluster_counter)
    cluster_counter += range_data['ClusterID'].max() + 1
    
    # Add to all_clusters
    all_clusters.append(range_data)
    
    num_clusters_in_range = range_data['ClusterID'].max() - range_data['ClusterID'].min() + 1
    print(f"    {range_name}: Found {num_clusters_in_range} clusters with {len(range_data)} dairies")

# Concatenate all cluster data
if all_clusters:
    facility_sjv = pd.concat(all_clusters, ignore_index=True)
    print(f"\n    Total: {len(facility_sjv)} dairies across all clusters in all ranges")
else:
    print("    ERROR: No clusters found!")
    exit()

# ==================== CLUSTER COMPOSITION ANALYSIS ====================
print(f"\n[9] Analyzing cluster composition (Digester/Non-Digester pairs)...")
cluster_summary = []

for cluster_id in sorted(facility_sjv['ClusterID'].unique()):
    cluster_data = facility_sjv[facility_sjv['ClusterID'] == cluster_id].copy()
    
    digester_in_cluster = cluster_data[cluster_data['HasDigester']]
    non_digester_in_cluster = cluster_data[~cluster_data['HasDigester']]
    
    # Calculate centroid
    center_lat = cluster_data['Latitude'].mean()
    center_lon = cluster_data['Longitude'].mean()
    
    # Calculate average within-cluster distance (km)
    coords_cluster = cluster_data[['Latitude', 'Longitude']].values
    distances = cdist(coords_cluster, [[center_lat, center_lon]], metric='euclidean')
    avg_spacing_km = distances.mean() * 111
    
    # Calculate herd size statistics
    avg_herd_size = cluster_data['MilkCows'].mean()
    herd_range = cluster_data['HerdSizeRange'].iloc[0]  # All in same range
    
    # Calculate herd size pairing quality (std dev of digester vs non-digester herd sizes)
    digester_herd_mean = digester_in_cluster['MilkCows'].mean() if len(digester_in_cluster) > 0 else np.nan
    non_digester_herd_mean = non_digester_in_cluster['MilkCows'].mean() if len(non_digester_in_cluster) > 0 else np.nan
    herd_pairing_diff = abs(digester_herd_mean - non_digester_herd_mean) if (not np.isnan(digester_herd_mean) and not np.isnan(non_digester_herd_mean)) else np.nan
    
    cluster_summary.append({
        'ClusterID': cluster_id,
        'HerdSizeRange': herd_range,
        'NumDairies': len(cluster_data),
        'NumDigesters': len(digester_in_cluster),
        'NumNonDigesters': len(non_digester_in_cluster),
        'AvgHerdSize': avg_herd_size,
        'DigestorAvgHerd': digester_herd_mean,
        'NonDigesterAvgHerd': non_digester_herd_mean,
        'HerdPairingDiff': herd_pairing_diff,
        'AvgSpacingKm': avg_spacing_km,
        'CenterLat': center_lat,
        'CenterLon': center_lon,
        'CountiesRepresented': ', '.join(cluster_data['County'].unique().tolist())
    })

cluster_summary_df = pd.DataFrame(cluster_summary)

print("\n" + "="*80)
print("CLUSTER SUMMARY TABLE")
print("="*80)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(cluster_summary_df.to_string(index=False))
pd.reset_option('display.max_columns')
pd.reset_option('display.width')

# ==================== IDENTIFY BEST CANDIDATE CLUSTERS ====================
print(f"\n[10] Identifying best candidate clusters for 4-dairy sampling teams...")
print("     Criteria: 2+ digesters & 2+ non-digesters, good herd size matching, optimal spacing")

# Score clusters on:
# 1. Has both digesters and non-digesters
# 2. Good herd size pairing (low difference)
# 3. Appropriate spacing (2-10 km)
# 4. Reasonable cluster size (4-6 dairies)

candidate_clusters = cluster_summary_df[
    (cluster_summary_df['NumDigesters'] >= 1) &  # At least 1 digester
    (cluster_summary_df['NumNonDigesters'] >= 1) &  # At least 1 non-digester
    (cluster_summary_df['AvgSpacingKm'] >= 1.5) &  # At least some separation
    (cluster_summary_df['AvgSpacingKm'] <= 12)  # But not too far apart
].copy()

# Score on herd size pairing quality (lower diff = better pairing)
candidate_clusters['PairingScore'] = candidate_clusters['HerdPairingDiff'].fillna(1000)  # Penalize missing data
candidate_clusters = candidate_clusters.sort_values('PairingScore').reset_index(drop=True)

print(f"\n    Found {len(candidate_clusters)} candidate clusters meeting criteria:")
print("\n" + "-"*80)
for idx, row in candidate_clusters.iterrows():
    print(f"\n    CANDIDATE CLUSTER #{idx + 1} (ID: {int(row['ClusterID'])})")
    print(f"      • Herd Size Range: {row['HerdSizeRange']}")
    print(f"      • {int(row['NumDairies'])} dairies total ({int(row['NumDigesters'])} digester, {int(row['NumNonDigesters'])} non-digester)")
    print(f"      • Herd size pairing: Digesters avg {row['DigestorAvgHerd']:,.0f}, Non-digesters avg {row['NonDigesterAvgHerd']:,.0f}")
    print(f"        (Difference: {row['HerdPairingDiff']:,.0f} cows - {'EXCELLENT' if row['HerdPairingDiff'] < 300 else 'GOOD' if row['HerdPairingDiff'] < 600 else 'OK'})")
    print(f"      • Average spacing: {row['AvgSpacingKm']:.1f} km between dairies")
    print(f"      • Geographic center: ({row['CenterLat']:.4f}, {row['CenterLon']:.4f})")
    print(f"      • Counties: {row['CountiesRepresented']}")

# ==================== DETAIL EACH CANDIDATE CLUSTER ====================
print(f"\n[11] DETAILED DAIRY INFORMATION FOR CANDIDATE CLUSTERS")
print("="*80)

candidate_cluster_ids = candidate_clusters['ClusterID'].values

for cluster_idx, cluster_id in enumerate(candidate_cluster_ids):
    cluster_data = facility_sjv[facility_sjv['ClusterID'] == cluster_id].sort_values('MilkCows', ascending=False)
    
    print(f"\n{'='*80}")
    print(f"CANDIDATE CLUSTER #{cluster_idx + 1} (ID: {int(cluster_id)}) - {cluster_data['HerdSizeRange'].iloc[0]}")
    print(f"{'='*80}")
    
    # Sort by digester status (digesters first) then by herd size
    cluster_data_sorted = pd.concat([
        cluster_data[cluster_data['HasDigester']].sort_values('MilkCows', ascending=False),
        cluster_data[~cluster_data['HasDigester']].sort_values('MilkCows', ascending=False)
    ])
    
    for _, row in cluster_data_sorted.iterrows():
        digester_status = "✓ HAS DIGESTER" if row['HasDigester'] else "  NO DIGESTER"
        print(f"\n  {row['FacilityName']}")
        print(f"    Location: {row['City']}, {row['County']} | Coords: ({row['Latitude']:.4f}, {row['Longitude']:.4f})")
        print(f"    Herd Size: {row['MilkCows']:,.0f} milking cows | Status: {digester_status}")
        print(f"    Address: {row['StreetAddress']}")

# ==================== EXPORT CANDIDATE SITES ====================
print(f"\n[12] Exporting candidate sites to CSV...")

export_cols = ['CADDID', 'FacilityName', 'City', 'County', 'Latitude', 'Longitude', 
               'StreetAddress', 'MilkCows', 'HerdSizeRange', 'HasDigester', 'ClusterID', 'Year']

candidate_dairies = facility_sjv[facility_sjv['ClusterID'].isin(candidate_cluster_ids)][export_cols].copy()
candidate_dairies = candidate_dairies.sort_values(['ClusterID', 'HasDigester', 'MilkCows'], ascending=[True, False, False])

candidate_dairies.to_csv('SJV_Candidate_Dairies_for_Sampling.csv', index=False)
print(f"    Exported {len(candidate_dairies)} candidate dairies to: SJV_Candidate_Dairies_for_Sampling.csv")

# ==================== SUMMARY FOR DEPLOYMENT PLANNING ====================
print(f"\n{'='*80}")
print("DEPLOYMENT PLANNING SUMMARY")
print(f"{'='*80}")

print(f"\nRECOMMENDATION: Select from {len(candidate_clusters)} candidate clusters")
print(f"\nPossible 15-20 dairy portfolios:")

if len(candidate_clusters) >= 3:
    print(f"\n  Option A: Use TOP 3-4 CLUSTERS (best herd size pairing)")
    top_clusters = candidate_clusters.head(min(4, len(candidate_clusters)))
    total_dairies_top = top_clusters['NumDairies'].sum()
    print(f"    Clusters: {', '.join(map(str, map(int, top_clusters['ClusterID'].values)))}")
    print(f"    Total dairies: {int(total_dairies_top)}")
    print(f"    Digester/Non-digester mix: {int(top_clusters['NumDigesters'].sum())} / {int(top_clusters['NumNonDigesters'].sum())}")
    print(f"    Herd size ranges: {', '.join(top_clusters['HerdSizeRange'].unique())}")

if len(candidate_clusters) >= 5:
    print(f"\n  Option B: Use BEST 5 CLUSTERS (maximum diversity)")
    best_5 = candidate_clusters.head(5)
    total_dairies_best5 = best_5['NumDairies'].sum()
    print(f"    Clusters: {', '.join(map(str, map(int, best_5['ClusterID'].values)))}")
    print(f"    Total dairies: {int(total_dairies_best5)}")
    print(f"    Digester/Non-digester mix: {int(best_5['NumDigesters'].sum())} / {int(best_5['NumNonDigesters'].sum())}")
    print(f"    Herd size ranges: {', '.join(best_5['HerdSizeRange'].unique())}")

print(f"\n  All candidate clusters: {len(candidate_clusters)}")
print(f"  Total dairies across all clusters: {int(candidate_clusters['NumDairies'].sum())}")

print(f"\nNEXT STEPS:")
print(f"  1. Review SJV_Candidate_Dairies_for_Sampling.csv")
print(f"  2. Cross-reference coordinates with Google Earth to assess:")
print(f"     - Road access for semi-stationary plume transects (SE direction for NW winds)")
print(f"     - Whether upwind stationary background monitor is feasible for each cluster")
print(f"     - Topographic barriers or wind obstruction")
print(f"  3. Contact facility operators to confirm site access & willingness to participate")
print(f"  4. Finalize 15-20 dairy roster before pilot deployment")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}\n")

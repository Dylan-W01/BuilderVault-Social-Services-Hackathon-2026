# From BuilderVault
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from pathlib import Path
from difflib import SequenceMatcher

pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 200)

DATA_DIR = Path('./data/raw')

expected_files = [
    'organizations.parquet',
    'clients.parquet',
    'referrals.parquet',
    'service_encounters.parquet',
    'consent_records.parquet',
    'data_sharing_agreements.parquet',
    'duplicate_flags.parquet',
]

missing = [f for f in expected_files if not (DATA_DIR / f).exists()]
if missing:
    print('Missing data files:', missing)
    print('Run: python ../generator/generate.py')
else:
    print(f'All {len(expected_files)} data files present in {DATA_DIR.resolve()}')

orgs = pd.read_parquet(DATA_DIR / 'organizations.parquet')
clients = pd.read_parquet(DATA_DIR / 'clients.parquet')
referrals = pd.read_parquet(DATA_DIR / 'referrals.parquet')
encounters = pd.read_parquet(DATA_DIR / 'service_encounters.parquet')
consent = pd.read_parquet(DATA_DIR / 'consent_records.parquet')
dsa = pd.read_parquet(DATA_DIR / 'data_sharing_agreements.parquet')
dup_flags = pd.read_parquet(DATA_DIR / 'duplicate_flags.parquet')

tables = {
    'orgs': orgs,
    'clients': clients,
    'referrals': referrals,
    'encounters': encounters,
    'consent': consent,
    'dsa': dsa,
    'duplicate_flags': dup_flags,
}

for name, df in tables.items():
    print(f'{name:20s} shape = {df.shape}')

rows = []
for name, df in tables.items():
    for col in df.columns:
        nulls = df[col].isna().sum()
        rows.append({
            'table': name,
            'column': col,
            'dtype': str(df[col].dtype),
            'null_count': int(nulls),
            'null_pct': round(100.0 * nulls / max(len(df), 1), 2),
        })

null_summary = pd.DataFrame(rows)
print('Top 15 columns by null percentage:')
null_summary.sort_values('null_pct', ascending=False).head(15)

# rows = []
# for name, df in tables.items():
#     print(f'\n--- {name} ---')
#     for col in df.columns:
#         print(col)
expected_schema = {
    'orgs': ['org_id', 'org_name', 'org_type', 'address_city', 'service_taxonomy_code'],
    'clients': ['client_id', 'first_name', 'last_name', 'dob', 'ocap_protected',
                'current_consent_id', 'indigenous_identity', 'assessment_acuity_level'],
    'referrals': ['referral_id', 'client_id', 'referring_org_id', 'receiving_org_id',
                  'status', 'submitted_at', 'decision_at', 'completed_at'],
    'encounters': ['encounter_id', 'client_id', 'org_id', 'encounter_type',
                   'encounter_start', 'encounter_end'],
    'consent': ['consent_id', 'client_id', 'collecting_org_id', 'status',
                'sharing_scope_type', 'purpose_codes', 'effective_date', 'expiry_date'],
    'dsa': ['dsa_id', 'dsa_name', 'dsa_type', 'effective_date', 'expiry_date'],
    'duplicate_flags': ['duplicate_flag_id', 'client_id_primary', 'client_id_secondary',
                        'match_score', 'review_status'],
}

all_pass = True
for name, cols in expected_schema.items():
    df = tables[name]
    missing_cols = [c for c in cols if c not in df.columns]
    status = 'OK' if not missing_cols else 'FAIL'
    if missing_cols:
        all_pass = False
    print(f'[{status}] {name:20s} expected {len(cols):2d} cols, missing: {missing_cols or "none"}')

assert all_pass, 'Schema check failed. Regenerate the dataset or update expected_schema above.'
print('\nAll expected columns present.')

referring = orgs[['org_id', 'org_name', 'org_type']].rename(columns={
    'org_id': 'referring_org_id',
    'org_name': 'referring_org_name',
    'org_type': 'referring_org_type',
})

receiving = orgs[['org_id', 'org_name', 'org_type']].rename(columns={
    'org_id': 'receiving_org_id',
    'org_name': 'receiving_org_name',
    'org_type': 'receiving_org_type',
})

client_slim = clients[['client_id', 'first_name', 'last_name', 'dob',
                       'indigenous_identity', 'ocap_protected',
                       'assessment_acuity_level', 'current_consent_id']]

consent_slim = consent[['consent_id', 'status', 'sharing_scope_type',
                        'effective_date', 'expiry_date']].rename(columns={
    'consent_id': 'current_consent_id',
    'status': 'consent_status',
    'sharing_scope_type': 'consent_scope',
    'effective_date': 'consent_effective_at',
    'expiry_date': 'consent_expires_at',
})

referrals_enriched = (
    referrals
    .merge(client_slim, on='client_id', how='left')
    .merge(referring, on='referring_org_id', how='left')
    .merge(receiving, on='receiving_org_id', how='left')
    .merge(consent_slim, on='current_consent_id', how='left')
)

curated_cols = [
    'referral_id', 'client_id', 'first_name', 'last_name',
    'referring_org_name', 'referring_org_type',
    'receiving_org_name', 'receiving_org_type',
    'status', 'submitted_at', 'decision_at', 'completed_at',
    'assessment_acuity_level', 'indigenous_identity', 'ocap_protected',
    'consent_status', 'consent_scope', 'consent_expires_at',
]

print(f'referrals_enriched shape: {referrals_enriched.shape}')
referrals_enriched[curated_cols].head(10)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Referral status breakdown
status_counts = referrals['status'].value_counts()
axes[0, 0].bar(status_counts.index, status_counts.values, color='steelblue')
axes[0, 0].set_title('Referral status breakdown')
axes[0, 0].set_ylabel('count')
axes[0, 0].tick_params(axis='x', rotation=30)

# 2. Referral volume by referring org type
by_ref_type = referrals_enriched['referring_org_type'].value_counts()
axes[0, 1].bar(by_ref_type.index, by_ref_type.values, color='darkorange')
axes[0, 1].set_title('Referral volume by referring org type')
axes[0, 1].set_ylabel('count')
axes[0, 1].tick_params(axis='x', rotation=30)

# 3. Consent status mix
consent_mix = consent['status'].value_counts()
axes[0, 2].pie(consent_mix.values, labels=consent_mix.index, autopct='%1.1f%%',
               startangle=90)
axes[0, 2].set_title('Consent status mix')

# 4. Client acuity level distribution
acuity_counts = clients['assessment_acuity_level'].dropna().value_counts().sort_index()
axes[1, 0].bar(acuity_counts.index, acuity_counts.values,
               color='seagreen', edgecolor='white')
axes[1, 0].set_title('Client acuity level distribution')
axes[1, 0].set_xlabel('acuity level')
axes[1, 0].set_ylabel('clients')
axes[1, 0].tick_params(axis='x', rotation=30)

# 5. Referral lifecycle time (submitted to completed or declined)
closed = referrals_enriched[referrals_enriched['status'].isin(
    ['completed', 'declined'])].copy()
closed['submitted_at'] = pd.to_datetime(closed['submitted_at'], errors='coerce')
closed['decision_at'] = pd.to_datetime(closed['decision_at'], errors='coerce')
closed['lifecycle_days'] = (closed['decision_at'] - closed['submitted_at']
                             ).dt.total_seconds() / 86400.0
lifecycle = closed['lifecycle_days'].dropna()
lifecycle = lifecycle[lifecycle > 0]
axes[1, 1].hist(lifecycle, bins=40, color='purple', edgecolor='white')
axes[1, 1].set_yscale('log')
axes[1, 1].set_title('Referral lifecycle (days, log scale)')
axes[1, 1].set_xlabel('days submitted to decision')

# 6. Summary card
axes[1, 2].axis('off')
summary_text = (
    f"Totals\n"
    f"orgs:       {len(orgs):>6,}\n"
    f"clients:    {len(clients):>6,}\n"
    f"referrals:  {len(referrals):>6,}\n"
    f"encounters: {len(encounters):>6,}\n"
    f"consent:    {len(consent):>6,}\n"
    f"dsa:        {len(dsa):>6,}\n"
    f"dup_flags:  {len(dup_flags):>6,}"
)
axes[1, 2].text(0.05, 0.5, summary_text, fontfamily='monospace', fontsize=12,
                verticalalignment='center')

plt.tight_layout()
# plt.show()

# Baseline: rule-based duplicate detector using Soundex-style bucketing + string similarity

# Converts a last name toa  4-character phonetic code
def soundex(name):
    if not isinstance(name, str) or not name:
        return '0000'
    name = name.upper()
    first = name[0]
    mapping = {'B': '1', 'F': '1', 'P': '1', 'V': '1',
               'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
               'D': '3', 'T': '3',
               'L': '4',
               'M': '5', 'N': '5',
               'R': '6'}
    tail = ''.join(mapping.get(c, '') for c in name[1:])
    collapsed = ''
    prev = ''
    for c in tail:
        if c != prev:
            collapsed += c
        prev = c
    return (first + collapsed + '000')[:4]

# Takes two strings and returns a number from 0.0 to 1.0, representing similarity.
def sim(a, b):
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Applies soundex to every last name
work = clients[['client_id', 'first_name', 'last_name', 'dob']].copy()
work['last_soundex'] = work['last_name'].apply(soundex)

pairs = []
# Groups clients by similar last name
for _, bucket in work.groupby('last_soundex'):
    # Skips too short or too long buckets
    if len(bucket) < 2 or len(bucket) > 80:
        continue
        # Converts bucket to dictionary
    recs = bucket.to_dict('records')
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            # Change this from before to after
            first_sim = sim(a['first_name'], b['first_name'])
            # TODO: change for a better format
            dob_match = 1.0 if pd.notna(a['dob']) and pd.notna(b['dob']) and a['dob'] == b['dob'] else 0.0
            score = 0.7 * first_sim + 0.3 * dob_match
            if score >= 0.75:
                pairs.append({
                    'client_id_a': a['client_id'],
                    'client_id_b': b['client_id'],
                    'score': round(score, 3),
                })

baseline = pd.DataFrame(pairs)
print(f'Baseline predicted {len(baseline)} duplicate pairs')


# Normalize pair direction so (a,b) == (b,a) for join
def keyify(row, a_col, b_col):
    a, b = row[a_col], row[b_col]
    return tuple(sorted([a, b]))


gt = dup_flags.copy()
gt['pair_key'] = gt.apply(lambda r: keyify(r, 'client_id_primary', 'client_id_secondary'), axis=1)
baseline['pair_key'] = baseline.apply(lambda r: keyify(r, 'client_id_a', 'client_id_b'), axis=1)

gt_true_keys = set(gt[gt['review_status'] == 'confirmed_duplicate']['pair_key'])
gt_all_keys = set(gt['pair_key'])
pred_keys = set(baseline['pair_key'])

tp = len(pred_keys & gt_true_keys)
fp = len(pred_keys - gt_true_keys)
fn = len(gt_true_keys - pred_keys)
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-9)

print(f'\nBaseline vs ground truth:')
print(f'  true positives:  {tp}')
print(f'  false positives: {fp}')
print(f'  false negatives: {fn}')
print(f'  precision: {precision:.3f}')
print(f'  recall:    {recall:.3f}')
print(f'  f1:        {f1:.3f}')
print('\nPrevious model version.')

######### Modified code by Dylan

# Baseline: rule-based duplicate detector using Soundex-style bucketing + string similarity

from dateutil import parser as dateparser
import jellyfish
# Documentation: https://github.com/life4/textdistance
import textdistance
# Converts a last name toa  4-character phonetic code

def soundex(name):
    if not isinstance(name, str) or not name:
        return '0000'
    name = name.upper()
    first = name[0]
    mapping = {'B': '1', 'F': '1', 'P': '1', 'V': '1',
               'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
               'D': '3', 'T': '3',
               'L': '4',
               'M': '5', 'N': '5',
               'R': '6'
               # ,'A': '7', 'E': '7', 'H':'7','I': '7', 'O': '7', 'U': '7', 'W': '7', 'Y': '7'
              } # originally missing A, E, H, I, O, U, W, Y, 
    tail = ''.join(mapping.get(c, '') for c in name[1:])
    collapsed = ''
    prev = ''
    for c in tail:
        if c != prev:
            collapsed += c
        prev = c
    return (first + collapsed + '000')[:4]

# Takes two strings and returns a number from 0.0 to 1.0, representing similarity.
def sim(a, b):
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    return textdistance.editex.normalized_similarity(a.lower(), b.lower())

# Applies soundex to every last name
def soundex_sorted(name):
    code = soundex(name)
    return code[0] + ''.join(sorted(code[1:])) #keeps first character, sorts remaining digits in ascending order.

work = clients[['client_id', 'first_name', 'last_name', 'aliases','dob']].copy()
work['last_soundex'] = work['last_name'].apply(soundex_sorted)
pairs = []
filtered = []
# Groups clients by similar last name
for _, bucket in work.groupby('last_soundex'):

    if len(bucket) < 2 or len(bucket) > 80:
        continue
        # Converts bucket to dictionary
    recs = bucket.to_dict('records')
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]

            first_sim = sim(a['first_name'], b['first_name']) 
            last_sim = sim(a['last_name'], b['last_name'])
            if first_sim < 0.6:
                p1 = 0
                p2 = 0
                if pd.notna(b['aliases']):
                    p1 = (sim(a['first_name'], b['aliases']))
                    
                if pd.notna(a['aliases']):
                    p2 = (sim(a['aliases'], b['first_name']))
                
                first_sim = max(p1, p2)
            
            try:
                if pd.notna(a['dob']) and pd.notna(b['dob']):
                    a_date = dateparser.parse(a['dob'], dayfirst=True).date()
                    b_date = dateparser.parse(b['dob'], dayfirst=True).date()
                    if a_date == b_date:
                        dob_match = 1.0
                    else: dob_match = 0.0
                else:
                    dob_match = 0.0
            except Exception:
                dob_match = 0.0
                
            score = 0.35 * first_sim + 0.35*last_sim + 0.3 * dob_match
            try:
                if pd.notna(a['dob']) and pd.notna(b['dob']) and dob_match == 0.0: score *= 0.7
            except Exception:
                dob_match = 0.0
            
            #Note this only consists of confirmed positives. 
            if score >= 0.56 and first_sim >= 0.5:
                pairs.append({
                    'client_id_a': a['client_id'],
                    'client_id_b': b['client_id'],
                    'score': round(score, 3),
                })
            else:
                 filtered.append({
                    'client_id_a': a['client_id'],
                    'client_id_b': b['client_id'],
                    'score': round(score, 3),
                })

baseline = pd.DataFrame(pairs)
print(f'Baseline predicted {len(baseline)} duplicate pairs')


# Normalize pair direction so (a,b) == (b,a) for join
def keyify(row, a_col, b_col):
    a, b = row[a_col], row[b_col]
    return tuple(sorted([a, b]))


gt = dup_flags.copy()
gt['pair_key'] = gt.apply(lambda r: keyify(r, 'client_id_primary', 'client_id_secondary'), axis=1)
baseline['pair_key'] = baseline.apply(lambda r: keyify(r, 'client_id_a', 'client_id_b'), axis=1) # PROBLEM HERE

gt_true_keys = set(gt[gt['review_status'] == 'confirmed_duplicate']['pair_key'])
gt_all_keys = set(gt['pair_key'])
pred_keys = set(baseline['pair_key'])

tp = len(pred_keys & gt_true_keys)
fp = len(pred_keys - gt_true_keys)
fn = len(gt_true_keys - pred_keys)
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 1e-9)

print(f'\nNew duplicate detector:')
print(f'  true positives:  {tp}')
print(f'  false positives: {fp}')
print(f'  false negatives: {fn}')
print(f'  precision: {precision:.3f}')
print(f'  recall:    {recall:.3f}')
print(f'  f1:        {f1:.3f}')


#########

# See what the algorithm missed
# confirmed = dup_flags[dup_flags['review_status'].isin(['confirmed_duplicate'])]
# cols = ['client_id', 'primary_org_id', 'first_name', 'last_name', 'middle_name', 'aliases', 'dob', 'phone', 'email']
# fp_keys = pred_keys - gt_true_keys
# fn_keys = gt_true_keys - pred_keys
# # Load your files
# clients_subset = clients[cols]
# for x in fn_keys:
#     id_a, id_b = x
#     pair = clients_subset[clients_subset['client_id'].isin([id_a, id_b])]
#     print(pair.to_string(index=False))
    
#     print('-' * 60)
    
# This code allows you to see all possible duplicates
"""
confirmed = dup_flags[dup_flags['review_status'].isin(['confirmed_duplicate'])]
cols = ['client_id', 'primary_org_id', 'first_name', 'last_name', 'middle_name', 'aliases', 'dob', 'phone', 'email']
fp_keys = pred_keys - gt_true_keys
fn_keys = gt_true_keys - pred_keys
# Load your files
clients_subset = clients[cols]
for x in fn_keys:
    id_a, id_b = x
    pair = clients_subset[clients_subset['client_id'].isin([id_a, id_b])]
    print(pair.to_string(index=False))
    
    print('-' * 60)
"""

# Generated by BuilderVault
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) Confusion matrix heatmap for baseline vs ground truth
tn_approx = len(gt_all_keys) - tp - fp - fn
cm = np.array([[tp, fn],
               [fp, max(tn_approx, 0)]])
im = axes[0].imshow(cm, cmap='Blues')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['predicted dup', 'predicted non-dup'])
axes[0].set_yticklabels(['actual dup', 'actual non-dup'])
axes[0].set_title('Baseline duplicate detector (within flagged pairs)')
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black',
                     fontsize=14)
plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

# (b) Top 10 orgs by consent gap rate
enc = encounters.merge(
    clients[['client_id', 'current_consent_id']], on='client_id', how='left'
).merge(
    consent[['consent_id', 'status', 'expiry_date']].rename(
        columns={'consent_id': 'current_consent_id'}),
    on='current_consent_id', how='left'
)
enc['encounter_start'] = pd.to_datetime(enc['encounter_start'], errors='coerce')
enc['expiry_date'] = pd.to_datetime(enc['expiry_date'], errors='coerce')
enc['has_gap'] = (
    enc['status'].isna()
    | (enc['status'].isin(['expired', 'withdrawn']))
    | ((enc['expiry_date'].notna()) & (enc['encounter_start'] > enc['expiry_date']))
)

gap_by_org = enc.merge(orgs[['org_id', 'org_name']], on='org_id', how='left')
gap_stats = gap_by_org.groupby('org_name').agg(
    encounters=('encounter_id', 'count'),
    gaps=('has_gap', 'sum'),
)
gap_stats['gap_rate'] = gap_stats['gaps'] / gap_stats['encounters']
gap_stats = gap_stats.sort_values('gap_rate', ascending=False).head(10)

axes[1].barh(gap_stats.index[::-1], gap_stats['gap_rate'].values[::-1], color='crimson')
axes[1].set_xlabel('consent gap rate (encounters without active matching consent)')
axes[1].set_title('Top 10 orgs by consent gap rate')
axes[1].set_xlim(0, 1)

plt.tight_layout()
# plt.show()

# Written by Dylan with the help of Gen AI

import json

# Enrich pairs with names so the UI can display them
client_lookup = clients.set_index('client_id')[['first_name', 'last_name', 'dob', 'ocap_protected',
                'current_consent_id', 'indigenous_identity', 'assessment_acuity_level']].to_dict('index')

queue = []
for p in pairs:
    a_info = client_lookup.get(p['client_id_a'], {})
    b_info = client_lookup.get(p['client_id_b'], {})
    a_dob = a_info.get('dob', '')
    try:
        if pd.notna(a_info.get('dob')):
            a_dob = str(dateparser.parse(a_info['dob'], dayfirst=True).date())
    
    except (Exception):
        a_dob = "N/A"
    
    
    b_dob = a_info.get('dob', '')
    try:
        if pd.notna(b_info.get('dob')):
            b_dob = str(dateparser.parse(b_info['dob'], dayfirst=True).date())
    except (Exception):
        b_dob = "N/A"

    queue.append({
        'client_id_a': p['client_id_a'],
        'name_a': f"{a_info.get('first_name', '')} {a_info.get('last_name', '')}".strip(),
        'dob_a': a_dob,
        'ocap_protected_a': str(a_info.get('ocap_protected', '')),
        'current_consent_id_a': str(a_info.get('current_consent_id', '')),
        'indigenous_identity_a': str(a_info.get('indigenous_identity', '')),
        'acuity_level_a': str(a_info.get('assessment_acuity_level', '')),

        'client_id_b': p['client_id_b'],
        'name_b': f"{b_info.get('first_name', '')} {b_info.get('last_name', '')}".strip(),
        'dob_b': b_dob,
        'ocap_protected_b': str(b_info.get('ocap_protected', '')),
        'current_consent_id_b': str(b_info.get('current_consent_id', '')),
        'indigenous_identity_b': str(b_info.get('indigenous_identity', '')),
        'acuity_level_b': str(b_info.get('assessment_acuity_level', '')),

        'score': p['score'],
    })



# Sort highest confidence first
queue.sort(key=lambda x: x['score'], reverse=True)

with open('queue.json', 'w') as f:
    json.dump(queue, f, indent=2)

print(f'Wrote {len(queue)} pairs to queue.json')

queue2 = []
for p in filtered:
    c_info = client_lookup.get(p['client_id_a'], {})
    d_info = client_lookup.get(p['client_id_b'], {})
    
    c_dob = ''
    try:
        raw = c_info.get('dob')
        if raw and pd.notna(raw):
            c_dob = str(dateparser.parse(str(raw), dayfirst=True).date())
    except Exception:
        c_dob = 'N/A'

    d_dob = ''
    try:
        raw = d_info.get('dob')
        if raw and pd.notna(raw):
            d_dob = str(dateparser.parse(str(raw), dayfirst=True).date())
    except Exception:
        d_dob = 'N/A'

    queue2.append({
        'client_id_a': p['client_id_a'],
        
        'name_a': f"{c_info.get('first_name', '')} {c_info.get('last_name', '')}".strip(),
        'dob_a': c_dob,
        'ocap_protected_a': str(c_info.get('ocap_protected', '')),
        'current_consent_id_a': str(c_info.get('current_consent_id', '')),
        'indigenous_identity_a': str(c_info.get('indigenous_identity', '')),
        'acuity_level_a': str(c_info.get('assessment_acuity_level', '')),

        'client_id_b': p['client_id_b'],
        'name_b': f"{d_info.get('first_name', '')} {d_info.get('last_name', '')}".strip(),
        'dob_b': d_dob,
        'ocap_protected_b': str(d_info.get('ocap_protected', '')),
        'current_consent_id_b': str(d_info.get('current_consent_id', '')),
        'indigenous_identity_b': str(d_info.get('indigenous_identity', '')),
        'acuity_level_b': str(d_info.get('assessment_acuity_level', '')),

        'score': p['score'],
    })

# Sort highest confidence first


queue2.sort(key=lambda x: x['score'], reverse=True)

with open('unsorted.json', 'w') as f:
    json.dump(queue2, f, indent=2)

print(f'Wrote {len(queue2)} pairs to queue.json')
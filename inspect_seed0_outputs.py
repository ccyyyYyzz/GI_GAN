import json
import pandas as pd
from pathlib import Path
root=Path('outputs/compatibility/measurement_conditioned_vqgan/anchor_initialized_seed0_hashclean/reports')
df=pd.read_csv(root/'final_dev_per_image.csv')
print('columns', df.columns.tolist())
print(df.head(3).to_string())
print('methods', df['method'].unique()[:30])
print(df[['method','beta']].drop_duplicates().head(30).to_string())
for name in ['gate_report.json','stage0_decision.json','refiner_manifests.json','summary.json']:
    p=root/name
    print('\n###', name)
    print(p.read_text(encoding='utf-8')[:3000])
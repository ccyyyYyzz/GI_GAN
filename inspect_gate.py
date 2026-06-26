import json
from pathlib import Path
p=Path('outputs/compatibility/measurement_conditioned_vqgan/anchor_initialized_seed0_hashclean/reports/gate_report.json')
g=json.loads(p.read_text())
print(g['selected_betas'].keys())
print(json.dumps(g['selected_betas'],indent=2)[:4000])
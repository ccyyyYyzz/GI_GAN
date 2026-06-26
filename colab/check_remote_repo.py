from pathlib import Path
p=Path('/content/repo')
print('repo_exists', p.exists(), 'entries', len(list(p.iterdir())) if p.exists() else 0)
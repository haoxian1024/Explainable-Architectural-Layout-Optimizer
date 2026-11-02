"""Run every supplied input, reconstruct/validate output, and render PNGs."""
import json
from pathlib import Path
from solver import solve,plot
root=Path(__file__).resolve().parent
(root/'results').mkdir(exist_ok=True)
for path in sorted((root/'examples').glob('*.json')):
    data=json.loads(path.read_text(encoding='utf-8'))
    result=solve(data)
    target=root/'results'/path.name
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    plot(data,result,target.with_suffix('.png'))
    print(path.name,result['status'],result.get('wall_count'),flush=True)

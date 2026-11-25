import csv, json, math, re, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'inputs'
results=json.loads((DATA/'results.json').read_text(encoding='utf-8'))['rounds']['r1']
errors=[]; checks=0

def parse_stat(value):
    s=str(value).strip().replace('%','').replace('−','-')
    if s in {'','–','-'}: return None,None
    m=re.match(r'^\s*([+-]?\d+(?:\.\d+)?)\s*(?:±\s*(\d+(?:\.\d+)?))?',s)
    if not m: raise ValueError(f'Cannot parse {value!r}')
    return float(m.group(1)), float(m.group(2)) if m.group(2) is not None else None

def eq(label, actual, expected, tol=.011):
    global checks
    checks+=1
    if actual is None or expected is None: return
    if abs(float(actual)-float(expected))>tol: errors.append(f'{label}: results.json={actual}, report={expected}')

metric_cols={'img AUROC':'img_auroc','pix AUROC':'pix_auroc','AUPRO':'aupro'}
with (DATA/'task4_r1/report/paper_inputs/table_a_main.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        t=r['target']; model=r['model']
        if model=='R0 (paper)': box=results.get('r0_published',{}).get(t,{})
        elif model=='R0 (reproduced)': box=results['r0_reproduced'].get(t,{})
        else:
            cfg='mh' if model=='M-h' else 'mh_ctrl'
            box=results['configs'][cfg][t]
        for col,m in metric_cols.items():
            value,std=parse_stat(r[col])
            if value is None: continue
            if model=='R0 (paper)': actual=box.get(m)
            elif model=='R0 (reproduced)': actual=box.get(m)
            else:
                actual=box[m]['mean']; actual_std=box[m]['std']
                eq(f'{t}/{model}/{col}/std',actual_std,std)
            eq(f'{t}/{model}/{col}',actual,value)

with (DATA/'task4_r1/report/paper_inputs/table_b_maskfree.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        t=r['target']; config='m' if r['model'].startswith('M') else 'b0'
        box=results['configs'][config][t]
        for col,m in metric_cols.items():
            value,std=parse_stat(r[col])
            if value is None: continue
            actual=box[m]['mean']; eq(f'{t}/{r["model"]}/{col}',actual,value)
            if std is not None: eq(f'{t}/{r["model"]}/{col}/std',box[m]['std'],std)
        ret=results['retention'][config][t]
        ret_value,_=parse_stat(r['retention (pix−50)/(R0−50)'])
        ratio_value,_=parse_stat(r['AUPRO / R0 AUPRO'])
        eq(f'{t}/{config}/retention_seed0',ret['seed0'],ret_value,tol=.0011)
        eq(f'{t}/{config}/aupro_ratio_seed0',ret['aupro_ratio_seed0'],ratio_value,tol=.0011)

with (DATA/'task4_r1/report/paper_inputs/table_a_deltas.csv').open(encoding='utf-8-sig',newline='') as f:
    paired={(x['target'],int(x['seed'])):x for x in results['paired']['mh-mh_ctrl']['per_pair']}
    for r in csv.DictReader(f):
        p=paired[(r['target'],int(r['seed']))]
        for col,m in [('Δ img AUROC','img_auroc'),('Δ pix AUROC','pix_auroc'),('Δ AUPRO','aupro')]:
            value,_=parse_stat(r[col]); eq(f'{r["target"]}/seed{r["seed"]}/{col}',p[m],value)

if errors:
    print('\n'.join(errors)); sys.exit(1)
print(f'PASS: {checks} values checked against round 1 report tables (tolerance 0.01).')

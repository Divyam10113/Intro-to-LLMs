import sys, json
sys.path.insert(0, "/home/claude/task3/logs/answers"); sys.path.insert(0, "/home/claude/task3/logs/answers/run2_qualsg")
from judge_lib import write_verdicts; from assess import A
RA = ["A1-g1-o6","A1-g2-o6","A1-g2-o4","A1-g1-o3","A1-g2-o2","A1-g1-o1","A1-g2-o1","A1-g1-o2","A1-g2-o3","A1-g0-C2","A1-g2-o5","A1-g1-o4"]
RB = ["B1-g1-o2","B1-g2-o1","B1-g0-C5","B1-g2-o3","B1-g1-o3","B1-g2-o6","B1-g0-C1","B1-g2-o4","B1-g1-o4","B1-g2-o5","B1-g1-o6","B1-g2-o2"]
out = "/home/claude/task3/logs/answers/run2_qualsg/"
write_verdicts(A, RA, out+"tA.json"); write_verdicts(A, RB, out+"tB.json")
a = {k:v for k,v in json.load(open(out+"tA.json")).items() if k.startswith("stage3/A1")}
b = {k:v for k,v in json.load(open(out+"tB.json")).items() if k.startswith("stage3/B1")}
json.dump({**a, **b}, open(out+sys.argv[1], "w"), indent=1); print(len(a), len(b))

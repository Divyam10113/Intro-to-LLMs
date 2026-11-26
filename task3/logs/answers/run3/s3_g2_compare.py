import sys, json
sys.path.insert(0, "/home/claude/task3/logs/answers"); from judge_lib import write_verdicts
D = "/home/claude/task3/logs/answers/run3/"
exec(open(D + 's3_assess.py').read())
write_verdicts(A, ["A1-g1-o2", "A1-g1-o6", "A1-g2-o2", "A1-g2-o4", "A1-g2-o1", "A1-g0-C1", "A1-g2-o5", "A1-g1-o1", "A1-g2-o6",
                   "A1-g0-C6", "A1-g2-o3", "A1-g1-o5"], D + "tmp/a.json")
write_verdicts(A, ["B1-g2-o1", "B1-g1-o3", "B1-g2-o4", "B1-g1-o5", "B1-g2-o3", "B1-g0-C1", "B1-g2-o6", "B1-g1-o4", "B1-g2-o2",
                   "B1-g1-o6", "B1-g2-o5", "B1-g1-o2"], D + "tmp/b.json")
a = {k: v for k, v in json.load(open(D + "tmp/a.json")).items() if k.startswith("stage3/A1")}
b = {k: v for k, v in json.load(open(D + "tmp/b.json")).items() if k.startswith("stage3/B1")}
json.dump({**a, **b}, open(D + "s3_g2_compare.json", "w"), indent=1)
print(len(a), len(b))

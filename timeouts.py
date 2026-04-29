import os
import sys
sys.path.append("..")
sys.path.append("../../..")
import json
import time

import math
import numpy as np
import torch

from utils import grid, evaluate
from enumerators import DecomposableEnum, DecomposableCount, MSD, MNCEnumerator, CircuitEnum, CircuitCount, topkCircuitCount, CE
from circuits.node import Node


# STEP 1 : Load the scores
cal_scores=torch.load("scores/cal_10405587.pt", map_location=torch.device("cpu"), weights_only=True).numpy()
cal_losses=torch.load("scores/cal_loss_10405587.pt", map_location=torch.device("cpu"), weights_only=True).numpy()
val_scores=torch.load("scores/val_10405587.pt", map_location=torch.device("cpu"), weights_only=True)

# STEP 2 : Compute non-conformity thresholds
print("~~~~~~~~~~~~ Compute thresholds enumeration ~~~~~~~~~~~~~~")
coverage_rates = 0.7+0.01*np.arange(20)
print("Coverage rates: ", coverage_rates)
numv = val_scores.shape[1]
print("Numer of variables: ", numv)
ncal = cal_losses.shape[0]
print("Size of calibration set: ", ncal)
indexes = np.floor(ncal*coverage_rates).astype('i')
print("Indexes: ", indexes)
cal_losses = np.sort(cal_losses, axis=0)
thresholds = cal_losses[indexes, 0] # [cal_losses[i,0] for i in indexes]
print("Thresholds: ", thresholds)

# STEP 3 : Initialize circuit
print("\n ~~~~~~~~~~~~ Compilation ~~~~~~~~~~~~~~")
sys.setrecursionlimit(10**6) # avoid recursion limit during compilation
n=12
s=time.time()
C = Node.AcyclicSimplePath(grid(n))
e=time.time()
print(f"The circuit was compiled in {e-s} seconds and has size {C.size()}")

# STEP 4 : Compute runtimes of confidence set enumeration
print("\n ~~~~~~~~~~~~ Start enumeration ~~~~~~~~~~~~~~")
tw=5
enumnames = ["mnc", "mod", "filtered"]
timeouts = {name:[] for name in enumnames}
runtimes = {name:[] for name in enumnames}
counts = {name:[] for name in enumnames}
msdsizes = []
runmsd=True
for (i,t) in enumerate(thresholds):
    print("\n ~~~~~~~~~~~~~~~")
    print("Coverage rate: ", coverage_rates[i])
    print("Threshold: ", t)

    msdsize = np.maximum(np.array([math.comb(numv, MSD(logits=s, thresh=t)) for s in val_scores]), 1).mean()
    msdsizes.append(msdsize)
    print("Mean size of MSD search space: ", msdsize)
    if runmsd:
        runtime, count, rto = evaluate(enumerator=MNCEnumerator, scores=val_scores, t=t, timewall=tw)
        if rto==1.0:
            runmsd=False
        timeouts["mnc"].append(rto)
        runtimes["mnc"].append(runtime)
        counts["mnc"].append(count)
        print("Ratio of timeouts of mnc: ", rto)
        print("Mean runtime of mnc: ", runtime)
        print("Mean count of mnc: ", count)
    else:
        print("MSD has reached 100% timouts")
        timeouts["mnc"].append(1.0)
        runtimes["mnc"].append(tw)
        counts["mnc"].append(0)

    runtime, count, rto = evaluate(enumerator=DecomposableEnum, scores=val_scores, t=t, timewall=tw)
    timeouts["mod"].append(rto)
    runtimes["mod"].append(runtime)
    counts["mod"].append(count)
    print("Ratio of timeouts of mod: ", rto)
    print("Mean runtime of mod: ", runtime)
    print("Mean count of mod: ", count)

    CircuitEnumf = lambda logits, thresh, timewall : CircuitEnum(logits=logits, thresh=thresh, ddnnf=C, timewall=timewall)
    runtime, count, rto = evaluate(enumerator=CircuitEnumf, scores=val_scores, t=t, timewall=tw)
    timeouts["filtered"].append(rto)
    runtimes["filtered"].append(runtime)
    counts["filtered"].append(count)
    print("Ratio of timeouts of filtered: ", rto)
    print("Mean runtime of filtered: ", runtime)
    print("Mean count of filtered: ", count)

with open('records.json') as f:
    data = json.load(f)
if tw is None:
    tw=0
data[tw] = {"timeouts":timeouts, "runtimes":runtimes, "counts":counts, "msdsizes":msdsizes}
with open('records.json', 'w') as f:
    json.dump(data, f)


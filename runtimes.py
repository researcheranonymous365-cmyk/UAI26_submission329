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
tw=2
enumnames = ["mnc", "dec", "filtered"]
runtimes = {name:[] for name in enumnames}
counts = {name:[] for name in enumnames}
run = {name:True for name in enumnames}

for (i,t) in enumerate(thresholds):
    print("\n ~~~~~~~~~~~~~~~")
    print("Coverage rate: ", coverage_rates[i])
    print("Threshold: ", t)

    if run["mnc"]:
        runtime, count, rto = evaluate(enumerator=MNCEnumerator, scores=val_scores, t=t, timewall=None)
        if runtime>=tw:
            run["mnc"]=False
        runtimes["mnc"].append(runtime)
        counts["mnc"].append(count)
        print("Mean runtime of mnc: ", runtime)
        print("Mean count of mnc: ", count)
    else:
        print("MSD average runtime is over timewall")

    if run["dec"]:
        runtime, count, rto = evaluate(enumerator=DecomposableEnum, scores=val_scores, t=t, timewall=None)
        if runtime>=tw:
            run["dec"]=False
        runtimes["dec"].append(runtime)
        counts["dec"].append(count)
        print("Mean runtime of decomposable enumeration: ", runtime)
        print("Mean count of decomposable enumeration: ", count)
    else:
        print("Decomposable enumeration average runtime is over timewall")

    if run["filtered"]:
        CircuitEnumf = lambda logits, thresh, timewall : CircuitEnum(logits=logits, thresh=thresh, ddnnf=C, timewall=timewall)
        runtime, count, rto = evaluate(enumerator=CircuitEnumf, scores=val_scores, t=t, timewall=None)
        if runtime>=tw:
            run["filtered"]=False
        runtimes["filtered"].append(runtime)
        counts["filtered"].append(count)
        print("Mean runtime of filtered enumeration: ", runtime)
        print("Mean count of filtered enumeration: ", count)
    else:
        print("Filtered enumeration average runtime is over timewall")

data = {"runtimes":runtimes, "counts":counts}
with open('runtimes.json', 'w') as f:
    json.dump(data, f)
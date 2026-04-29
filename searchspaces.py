import os
import sys
sys.path.append("..")
sys.path.append("../../..")
import json
import time

import math
import numpy as np
import torch

from utils import grid
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
print("\n ~~~~~~~~~~~~ Start counting ~~~~~~~~~~~~~~")
methods = ["mnc", "mod", "filtered"]
sizes = {name:[] for name in methods}
stds = {name:[] for name in methods}
for (i,t) in enumerate(thresholds[:15]):
    print("\n ~~~~~~~~~~~~~~~")
    print("Coverage rate: ", coverage_rates[i])
    print("Threshold: ", t)

    msdsizes = np.maximum(np.array([math.comb(numv, MSD(logits=s, thresh=t)) for s in val_scores]), 1)
    mean, std = msdsizes.mean(), msdsizes.std()
    sizes["mnc"].append(mean)
    stds["mnc"].append(std)
    print("Mean size of MSD search space: ", mean)
    
    dsizes = np.maximum(np.array([DecomposableCount(logits=s, thresh=t) for s in val_scores]), 1)
    mean, std = dsizes.mean(), dsizes.std()
    sizes["mod"].append(mean)
    stds["mod"].append(std)
    print(f"Mean size of decomposable search space: ", mean)
    
    fsizes = np.maximum(np.array([CircuitCount(logits=s, thresh=t, ddnnf=C) for s in val_scores]), 1)
    mean, std = fsizes.mean(), fsizes.std()
    sizes["filtered"].append(mean)
    stds["filtered"].append(std)
    print(f"Mean size of filtered search space: ", mean)

data = {"sizes":sizes, "stds":stds}
with open('sizes.json', 'w') as f:
    json.dump(data, f)
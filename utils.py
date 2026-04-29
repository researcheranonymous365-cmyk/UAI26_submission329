import numpy as np
import networkx as nx

import time
from concurrent.futures import TimeoutError
from functools import wraps
import time

def grid(n):
    g = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if i<n-1:
                g.add_edge((i, j), (i+1, j))
            if j<n-1:
                g.add_edge((i, j), (i, j+1))

    return g

def evaluate(enumerator, scores, t, timewall=None):
    runtimes=[]
    counts=[]
    timeouts = 0
    for score in scores:
        try:
            s=time.time()
            c=sum([1 for _ in enumerator(logits=score, thresh=t, timewall=timewall)])
            e=time.time()
            if timewall and e-s>=timewall:
                timeouts+=1
            else:
                runtimes.append(e-s)
                counts.append(c)

        except TimeoutError:
            timeouts+=1

    if len(runtimes)>0:
        runtime=np.array(runtimes).mean()
        count=np.maximum(np.array(counts), 1).mean()
        # ratio=np.divide(np.array(runtimes), np.maximum(np.array(counts), 1)).mean()
    else:
        runtime=timewall
        count=0
        # ratio=0
    rto = timeouts/len(scores)

    return runtime, count, rto
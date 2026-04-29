import time

import torch
import numpy as np

# from heapq import heappop, heappush
import queue

def CE(logit, b):
    logp = np.log(1/(1+np.exp(-logit)))
    if b==1:
        return - logp
    else:
        return logit - logp

def DecomposableCount(logits, thresh, f=CE, g=lambda x: x):
    """
    Count states in decreasing order of probabilities until a threshold is reached.

    Inputs:
        - logits : a tensor of log-probabilities for positive literals
        - thresh : the probability threshold that enumerated states must pass
        - f : a function that takes a logit and a binary value and outputs a non-conformity term
        - g : a function that takes the sum of non-conformity terms and computes the non-conformity score
    """
    count = 0
    n = logits.shape[0]
    scores = np.array([[f(logits[i], 0), f(logits[i], 1)] for i in range(n)])
    mins = np.min(scores, axis=1)
    maxs = np.max(scores, axis=1)
    diffs = maxs - mins
    vorder = np.argsort(diffs)
    argmins = np.argmin(scores, axis=1)
    mi = np.sum(mins)
    ma = np.sum(maxs)
    if g(ma)<=thresh:
        return 2**n
    elif g(mi)<=thresh:
        # queue = []
        # heappush(queue, (mi, ma, []))
        # while queue:
        q = queue.SimpleQueue()
        q.put(item=(mi, ma, []), block=False)
        while not(q.empty()):
            # mi, ma, prefix = heappop(queue)
            mi, ma, prefix = q.get(block=False)
            l=len(prefix)
            if l==n: # if the prefix is a full assignment,
                count +=1 # then yield the full assignment
            else: # otherwise, add to the queue the extensions which are below the threshold
                v=vorder[-l] # select the variable in the order that maximizes non-conformity diff at first
                prefix.append(argmins[v])
                if g(ma - diffs[v])<=thresh: # check if the max of the first extension is below threshold
                    count += 2**(n-l-1)
                else:
                    q.put(item=(mi, ma - diffs[v], prefix.copy()), block=False)
                    # heappush(queue, (mi, ma - diffs[v], prefix.copy())) # the minimum extension is automatically below threshold

                if g(mi + diffs[v])<=thresh: # check if the min of the second extension is below threshold
                    prefix[l] = 1 - argmins[v]
                    q.put(item=(mi + diffs[v], ma, prefix.copy()), block=False)
                    # heappush(queue, (mi + diffs[v], ma, prefix.copy())) # if so add it to the queue

    return count

def DecomposableEnum(logits, thresh, f=CE, g=lambda x: x, timewall=None):
    """
    Enumerate states in decreasing order of probabilities until a threshold is reached.

    Inputs:
        - logits : a tensor of log-probabilities for positive literals
        - thresh : the probability threshold that enumerated states must pass
        - f : a function that takes a logit and a binary value and outputs a non-conformity term
        - g : a function that takes the sum of non-conformity terms and computes the non-conformity score
    """
    start_time=time.time()
    n = logits.shape[0]
    scores = np.array([[f(logits[i], 0), f(logits[i], 1)] for i in range(n)])
    mins = np.min(scores, axis=1)
    maxs = np.max(scores, axis=1)
    diffs = maxs - mins
    argmins = np.argmin(scores, axis=1)
    m = np.sum(mins)
    if g(m)<=thresh:
        # queue = []
        # heappush(queue, (m, []))
        q = queue.SimpleQueue()
        q.put(item=(m, []), block=False)
        # while queue:
        while not(q.empty()):
            if timewall:
                if time.time()-start_time >= timewall:
                    raise TimeoutError("Enumeration timed out")
            # score, prefix = heappop(queue)
            score, prefix = q.get(block=False)
            l=len(prefix)
            if l==n: # if the prefix is a full assignment,
                yield torch.Tensor(prefix) # then yield the full assignment
            else: # otherwise, add to the queue the extensions which are below the threshold
                prefix.append(argmins[l])
                q.put(item=(score, prefix.copy()), block=False)
                # heappush(queue, (score, prefix.copy())) # the maximum extension is automatically below threshold

                if g(score + diffs[l])<=thresh: # check if the second extension is also below threshold
                    prefix[l] = 1 - argmins[l]
                    # heappush(queue, (score + diffs[l], prefix.copy())) # if so add it to the queue
                    q.put(item=(score + diffs[l], prefix.copy()), block=False)

def FlashlightEnum(logits, thresh, minimizer):
    """
    Enumerate states in decreasing order of probabilities until a threshold is reached.

    Inputs:
        - logits : a tensor of log-probabilities for positive literals
        - thresh : the probability threshold that enumerated states must pass
        - minimizer : a function that takes a prefix and computes the minimum non-conformity score of all its extensions
    """
    n = logits.shape[0]
    m = minimizer([], logits)
    if m<=thresh:
        # queue = []
        # heappush(queue, (m, []))
        q = queue.SimpleQueue()
        q.put(item=(m, []), block=False)
        # while queue:
        while not(q.empty()):
            # score, prefix = heappop(queue)
            score, prefix = q.get(block=False)
            l=len(prefix)
            if l==n: # if the prefix is a full assignment,
                yield torch.Tensor(prefix) # then yield the full assignment
            else: # otherwise, add to the queue the extensions which are below the threshold
                for b in [0,1]:
                    ext = prefix.copy().append(b) # create an extension of the prefix
                    m = minimizer(ext, logits)
                    if m<=thresh:
                        # heappush(queue, (m, ext)) # add the extension to the queue
                        q.put(item=(m, ext), block=False)


def FlashlightCount(logits, thresh, minimizer, maximizer):
    """
    Enumerate states in decreasing order of probabilities until a threshold is reached.

    Inputs:
        - logits : a tensor of log-probabilities for positive literals
        - thresh : the probability threshold that enumerated states must pass
        - minimizer : a function that takes a prefix and computes the minimum non-conformity score of all its extensions
        - maximizer : a function that takes a prefix and computes the maximum non-conformity score of all its extensions
    """
    n = logits.shape[0]
    mi, ma = minimizer([], logits), maximizer([], logits)
    count=0
    if ma<=thresh:
        return 2**n
    elif mi<=thresh:
        q = queue.SimpleQueue()
        q.put(item=[], block=False)
        while not(q.empty()):
            prefix = q.get(block=False)
            l=len(prefix)
            if l==n: # if the prefix is a full assignment,
                count+=1 # increment the count
            else: # otherwise, add to the queue the extensions which are below the threshold
                for b in [0,1]:
                    ext = prefix.copy().append(b) # create an extension of the prefix
                    mi, ma = minimizer(ext, logits), maximizer(ext, logits)
                    if ma<=thresh:
                        count+=2**(n-l)
                    elif mi<=thresh:
                        q.put(item=ext, block=False)


def triwhere(cond, c1, c2, c3, v0=0, v1=1):
    """
    Implements a triadique version of torch.where.

    Inputs:
        - cond: a tensor whose values decide for each cell from which tensor we pick the value
        - c1, c2, c3: three tensors from which we pick the value of the final tensor depending on cond
        - v0: the value we test to know when to pick from c1
        - v1: the value we test to know when to pick from c2

        All tensors cond c1, c2, c3 must be broadcastable to the same shape
    """

    return torch.where(torch.eq(cond, v0), c1, torch.where(torch.eq(cond, v1), c2, c3))

def topkCircuitCount(logits, thresh, ddnnf, k):
    """
    Enumerate instances in decreasing order of probabilities until a threshold is reached.

    Inputs:
        - logits : a tensor of log-probabilities for positive literals
        - thresh : the probability threshold that enumerated instances must pass
        - ddnnf : a dDNNF circuit
    """
    print("Starting topk circuit count")
    n = logits.shape[0]
    probs = torch.nn.functional.sigmoid(logits)
    maxp = torch.prod(torch.where(probs>=0.5, probs, 1-probs))
    if -torch.log(maxp).item() > thresh:
        return 0

    _, p = ddnnf.topk(probs=probs.unsqueeze(dim=0), k=k)
    count = torch.log(p).ge(-thresh).sum()
    
    while count==k:
        print("Doubling !")
        k=k*2
        _, p = ddnnf.topk(probs=probs.unsqueeze(dim=0), k=k)
        count = torch.log(p).ge(-thresh).sum()

    return count
    

def CircuitEnum(logits, thresh, ddnnf, timewall=None):
    """
    Enumerate filtered labelsets in the confidence set.

    Inputs:
        - logits : a tensor of log-probabilities for positive literals
        - thresh : the probability threshold that enumerated instances must pass
        - ddnnf : a dDNNF circuit
    """
    # print("Starting circuit enum")
    start_time=time.time()
    n = logits.shape[0]
    probs = torch.nn.functional.sigmoid(logits)
    vorder = torch.argsort(torch.abs(probs-0.5), descending=True)
    logprobs = torch.log(torch.stack([probs, 1-probs]))
    probs = probs.unsqueeze(dim=0)
    mins = logprobs.min(dim=0).values
    maxs = logprobs.max(dim=0).values
    s, p = ddnnf.mpe(probs=probs)
    mini, maxp  = s[0], p[0]
    if -torch.log(maxp)<=thresh:
        q = queue.SimpleQueue()
        q.put(item=(0, mini), block=False)
        trans = torch.sub(torch.tensor([[i >= j for j in range(n)] for i in range(n)], dtype=torch.int), 2*torch.eye(n))
        trans = torch.index_select(trans, 1, vorder)
        while not(q.empty()):
            if timewall:
                if time.time()-start_time >= timewall:
                    raise TimeoutError("Enumeration timed out")
            l, mini = q.get(block=False)
            yield mini
            trimini = 2*mini-1 # tri-valued version of the min instantiation
            new_prefixes = torch.mul(trimini, trans[l:]) # create new prefixes by flipping one bit, starting from l+1 up to n

            # test the last flip independently
            lastnll = -torch.sum(torch.where(torch.eq(new_prefixes[-1], 1), logprobs[0], logprobs[1]))
            if lastnll<=thresh:
                yield lastnll.ge(0)
            
            new_prefixes = new_prefixes[:-1] # remove the last prefix

            # first set aside prefixes whose min unconstrained extension is already above threshold
            minnll = -torch.sum(triwhere(new_prefixes, maxs, logprobs[0], logprobs[1]), dim=1)
            to_test = torch.nonzero(minnll.le(thresh)).flatten()

            # if there are still prefixes, test if they have a min constrained extension below threshold
            nbtest = to_test.shape[0]
            if nbtest>0:
                test_prefixes=torch.index_select(new_prefixes, dim=0, index=to_test)
                lens=n-torch.eq(test_prefixes,0).sum(dim=1)
                s, p = ddnnf.mpe(probs=probs, evidence=test_prefixes)
                for i in range(nbtest):
                    mini, maxp = s[i,:], p[i]
                    if maxp!=0 and -torch.log(maxp)<=thresh:
                        q.put(item=(lens[i], mini), block=False)


def CircuitCount(logits, thresh, ddnnf):
    """
    Counts the instances in the confidence set.

    Inputs:
        - logits : a tensor of log-probabilities for positive literals
        - thresh : the probability threshold that enumerated instances must pass
        - ddnnf : a dDNNF circuit
    """
    # print("Starting circuit count")
    n = logits.shape[0]
    probs = torch.nn.functional.sigmoid(logits)
    vorder = torch.argsort(torch.abs(probs-0.5), descending=True)
    probs = torch.stack([probs, 1-probs])
    logprobs = torch.log(probs)
    mins = logprobs.min(dim=0).values
    maxs = logprobs.max(dim=0).values
    s, p = ddnnf.mpe(probs=probs)
    maxi, maxp, mini  = s[0], p[0], s[1]
    minp = torch.prod(torch.where(mini.eq(1), probs[0], probs[1]))
    count=0
    if -torch.log(minp)<=thresh:
        return 2**n
    elif -torch.log(maxp)<=thresh:
        q = queue.SimpleQueue()
        count+=1
        q.put(item=(0, maxi), block=False)
        trans = torch.sub(torch.tensor([[i >= j for j in range(n)] for i in range(n)], dtype=torch.int), 2*torch.eye(n))
        trans = torch.index_select(trans, 1, vorder)
        while not(q.empty()):
            l, maxi = q.get(block=False)
            trimaxi = 2*maxi-1 # tri-valued version of the min instantiation
            new_prefixes = torch.mul(trimaxi, trans[l:]) # create new prefixes by flipping one bit, starting from l+1 up to n

            # test the last flip independently
            lastnll = -torch.sum(torch.where(torch.eq(new_prefixes[-1], 1), logprobs[0], logprobs[1]))
            if lastnll<=thresh:
                count+=1
            
            new_prefixes = new_prefixes[:-1] # remove the last prefix

            # first set aside prefixes whose max unconstrained extension is already below threshold
            to_count=[]
            maxnll = -torch.sum(triwhere(new_prefixes, mins, logprobs[0], logprobs[1]), dim=1)
            maxbelowt = torch.nonzero(maxnll.le(thresh)).flatten()
            if maxbelowt.shape[0]>0:
                to_count=[torch.index_select(new_prefixes, dim=0, index=maxbelowt)]
                new_prefixes = torch.index_select(new_prefixes, dim=0, index=torch.nonzero(~maxnll.le(thresh)).flatten())

            # then set aside prefixes whose min unconstrained extension is already above threshold
            minnll = -torch.sum(triwhere(new_prefixes, maxs, logprobs[0], logprobs[1]), dim=1)
            to_test = torch.nonzero(minnll.le(thresh)).flatten()

            # if there are still prefixes in between, test if they have a min (resp. max) constrained extension below threshold
            nbtest = to_test.shape[0]
            if nbtest>0:
                test_prefixes=torch.index_select(new_prefixes, dim=0, index=to_test)
                lens=n-torch.eq(test_prefixes,0).sum(dim=1)
                s, p = ddnnf.mpe(probs=probs, evidence=test_prefixes)
                for i in range(nbtest):
                    maxi, maxp, mini = s[2*i,:], p[2*i], s[2*i+1]
                    minp = torch.prod(torch.where(mini.eq(1), probs[0], probs[1]))
                    if minp!=0 and -torch.log(minp)<=thresh:
                        to_count.append(test_prefixes[i].unsqueeze(dim=0))
                    elif maxp!=0 and -torch.log(maxp)<=thresh:
                        count+=1
                        q.put(item=(lens[i], maxi), block=False)
            
            if len(to_count)>0:
                evidence=torch.cat(to_count, dim=0)
                count+=int(ddnnf.mc(evidence=evidence).sum())

    return count


def MSD(logits, thresh, f=CE):
    """
    Compute the maximum symetric difference such that any labelset with a higher symetric difference with the optimal labelset does not belong to the confidence set.

    Inputs:
        - probs : a tensor of probabilities for positive literals
        - thresh : the probability threshold that enumerated states must pass
        - f : a function that takes a logit and a binary value and outputs a non-conformity term
        - g : a function that takes the sum of non-conformity terms and computes the non-conformity score
    """
    n = logits.shape[0]
    scores = np.array([[f(logits[i], 0), f(logits[i], 1)] for i in range(n)])
    mins = np.min(scores, axis=1)
    maxs = np.max(scores, axis=1)
    diffs = maxs - mins
    sumodiffs = np.cumsum(np.sort(diffs)) # sums of ordered diffs
    # print("First sums of differences: ", sumodiffs[:10])
    m = np.sum(mins)
    # print("Minimal non conformity score: ", m)
    # print("Threshold: ", thresh)
    t = np.argmax(sumodiffs >= thresh - m) # compute the minmum symetric difference to contain all labelsets under the threshold

    return t

def MNCEnumerator(logits, thresh, f=CE, timewall=None):
    """
    Minimum Non-Conformity Change Enumeration of labelsets in decreasing order of probabilities until a threshold is reached.

    Inputs:
        - logits : a tensor of logits produced by the network
        - thresh : the probability threshold that enumerated states must pass
        - f : a function that takes a logit and a binary value and outputs a non-conformity term
        - g : a function that takes the sum of non-conformity terms and computes the non-conformity score
    """
    start_time=time.time()
    n = logits.shape[0]
    scores = np.array([[f(logits[i], 0), f(logits[i], 1)] for i in range(n)])
    mins = np.min(scores, axis=1)
    maxs = np.max(scores, axis=1)
    m = np.sum(mins)
    argmins = np.argmin(scores, axis=1)
    t = MSD(logits, thresh, f)
    q = queue.SimpleQueue()
    if m <= thresh:
        q.put(item=(0, []), block=False)
    while not(q.empty()):
        if timewall:
            if time.time()-start_time >= timewall:
                raise TimeoutError("Enumeration timed out")
        sd, prefix = q.get(block=False)
        l=len(prefix)
        if l==n: # if the prefix is a full assignment,
            selector = np.array(prefix).astype(bool)
            score = np.sum(np.where(selector, mins, maxs)) # compute the score of the labelset
            if score <= thresh: # if it's below the threshold
                yield np.where(selector, argmins, 1-argmins) # then yield the selected labelset
        else: # otherwise, add to the queue the extensions when their symetric difference is under t
            q.put(item=(sd, prefix+[1]), block=False) # add the optimal extension to the queue
            if sd+1 <= t:
                q.put(item=(sd+1, prefix+[0]), block=False) # add the sub-omptimal extension to the queue
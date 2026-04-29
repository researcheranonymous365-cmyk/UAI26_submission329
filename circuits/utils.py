import torch
from torch import Tensor
import torch.nn.functional as F

from heapq import heappop, heappush
import queue

def memo(func):
    '''
    A memoization decorator that stores the value of a static attribute of a node once it has been computed by the function.
    Allows to recursively call the function without computing the same static attribute several times for a given node.
    '''
    name=func.__name__
    def memo_func(self, *args, **kwargs):
        # print(self.attributes)
        # value = self.attributes.get(name, None)
        if name not in self.attributes:
            self.attributes[name] = func(self, *args, **kwargs)

        return self.attributes[name]
        # if not(value is None):
        #     return value
        # else:
        #     value = func(self, *args, **kwargs)
        #     self.attributes[name] = value
        #     return value

    return memo_func

def caching(postprocess=None):
    '''
    A parameterized caching decorator that stores the value of a dynamic attribute of a node once it has been computed by the function.
    Allows to recursively call the function without computing the same dynamic attribute several times for a node.
    Clears the cache so that new values can be stored in the cache during the next initial call.
    Applies a postprocessing function before returning the final output.
    '''
    def decorator(func):
        name=func.__name__
        def cached_func(self, *args, **kwargs):
            # First check it this is the first call
            first_call = kwargs.get("first_call", True)
            if first_call:
                kwargs["first_call"]=False

            output = self.cache.get(name, None)
            if not(output is None):
                return output
            else:
                output = func(self, *args, **kwargs)

            if first_call:
                self.clear_cache(key=name)
                if postprocess is not None:
                    output = postprocess(output)
            else:
                self.cache[name] = output

            return output

        return cached_func
    
    return decorator

def iter(clear=False, postprocess=None):
    '''
    A parameterized caching decorator that applies the function on each sub-node of the circuit and stores the value in a cache.
    '''

    def decorator(func):
        name=func.__name__
        def wrapper(self, *args, **kwargs):
            value=self.cache.get(name)
            if value is not None:
                return value
            else:
                for node in self.iter():
                    value = node.cache.get(name)
                    if value is None:
                        node.cache[name] = func(node, *args, **kwargs)

                value = self.cache[name]
                if postprocess:
                    value = postprocess(value)
                if clear:
                    self.clear_cache(name)

                return value
        
        return wrapper
    return decorator



def logsumexp(tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    with torch.no_grad():
        m, _ = torch.max(tensor, dim=dim, keepdim=True)
        m = m.masked_fill_(torch.isneginf(m), 0.)

    z = (tensor - m).exp_().sum(dim=dim, keepdim=True)
    mask = z == 0
    z = z.masked_fill_(mask, 1.).log_().add_(m)
    z = z.masked_fill_(mask, -float('inf'))

    if not keepdim:
        z = z.squeeze(dim=dim)
    return z

def logsum(logprobs):
    return torch.log(torch.sum(torch.exp(logprobs), dim=1))

# @torch.jit.script
def log1mexp(logprobs):
    return torch.log1p(-torch.exp(logprobs))


def enumerate_states(variables):
    if len(variables)==1:
        yield {variables[0]:1}
        yield {variables[0]:-1}
    else:
        for state in enumerate_states(variables[1:]):
            yield {**state, **{variables[0]:1}}
            yield {**state, **{variables[0]:-1}}


# top-k

def keep_topk(states, p, k, d=1, n=None):
    if p.shape[1] <= k:
        return states, p
    if n is None:
        n=states.shape[1]
    p, idx = torch.topk(p, k=k, dim=d)
    idx=idx.unsqueeze(dim=d).expand(-1,n,-1)
    states = torch.gather(states, dim=d+1, index=idx)
    return states, p

def smooth_topk(probs, statesn, pn, k, variables):
    bs, kn, n = probs.shape[0], pn.shape[1], statesn.shape[1]
    fix = [ind_topk(probs=probs[i], k=k, variables=variables) for i in range(bs)]
    fix_states = torch.stack([fix[i][0] for i in range(bs)], dim=0)
    fix_p = torch.stack([fix[i][1] for i in range(bs)], dim=0)
    kf=fix_p.shape[1]
    pn = (pn[:, :, None] * fix_p[:, None, :]).reshape(bs, kn*kf)
    statesn = (statesn[:, :, :, None] + fix_states[:, :, None, :]).reshape(bs, n, kn*kf)
    if kn*kf > k:
        statesn, pn = keep_topk(statesn, pn, k=k, n=n)

    return statesn, pn

def ind_topk(probs, k, vars):
    n=probs.shape[0]
    numv=len(vars)
    vars=torch.Tensor(vars)
    mask = torch.zeros(n)
    mask[vars]=1
    c=0
    topk = []
    p = []
    probs = probs[vars]
    vorder = torch.argsort(torch.abs(probs-0.5), descending=True)
    probs = torch.stack([1-probs, probs])
    maxs = probs.max(dim=0).values
    m = maxs.prod()
    q = queue.PriorityQueue()
    q.put(item=(m, torch.zeros(numv)), block=False)
    while c<k and not(q.empty()):
        m, prefix = q.get(block=False)
        l = torch.eq(0).sum()
        if l==0:
            c+=1
            topk.append(prefix)
            p.append(m)
        else:
            v=vorder[l]
            for b in [0,1]:
                prefix[v]=2*b-1
                nm=m*probs[b][v]/maxs[v]
                q.put(item=(nm, prefix), block=False)

    topk = torch.stack(topk, dim=1)
    p = torch.stack(p)

    return topk, p

# def ind_topk(probs, k=2, boolean=False, variables=None):
#     bs, n = probs.shape

#     if variables is None:
#         variables = list(range(n))

#     states = torch.zeros((bs, n, 1), dtype=torch.bool)
#     p = torch.ones((bs, 1))
#     for i in variables:
#         value = F.one_hot(torch.Tensor([i]).long(), num_classes=n).unsqueeze(-1).expand((bs, n, states.shape[2]))
#         states = torch.cat([torch.add(states, value), torch.add(states, -value)], dim=2)
#         p = torch.cat([torch.mul(p, probs[:, i].unsqueeze(-1)), torch.mul(p, 1-probs[:, i].unsqueeze(-1))], dim=1)
#         if states.shape[2] > k:
#             p, idx = torch.topk(p, k=k, dim=1)
#             idx=idx.unsqueeze(1).expand(-1, states.shape[1], k)
#             states = torch.gather(states, dim=2, index=idx)
#         else:
#             p, idx = torch.sort(p, dim=1, descending=True)
#             idx=idx.unsqueeze(1).expand(-1, states.shape[1], states.shape[2])
#             states = torch.gather(states, dim=2, index=idx)
    
#     if boolean:
#         states = states.ge(0)

#     return states, p

# enumeration

def enumerate(probs, thresh):
    n = probs.shape[0]
    maxs = torch.where(probs.ge(0.5), probs, 1-probs)
    maxs = torch.stack([torch.prod(maxs[j:]) for j in range(n)], dim=0)
    maxs = torch.concat([maxs, torch.ones(1)], dim=0)
    mpe = maxs[0]
    if mpe>=thresh:
        queue = [(mpe, torch.zeros(n))]
        while queue:
            m, gamma = heappop(queue)
            for j in range(int(gamma.abs().sum()), n):
                if probs[j]>=0.5:
                    p = m*(1-probs[j])*maxs[j+1]
                    if p>=thresh:
                        gamma[j]=-1
                        heappush(queue, (p, gamma.clone()))
                    gamma[j]=1
                else:
                    p = m*probs[j]*maxs[j+1]
                    if p>=thresh:
                        gamma[j]=1
                        heappush(queue, (p, gamma.clone()))
                    gamma[j]=-1
            yield gamma.ge(0)


#### SDD UTILS #########
# from pysdd.sdd import Vtree

# def right_linear_vtree(n):
#     for i in range(1, n):
#         if i==1:
#             right=Vtree.leaf_node(var=i)
#         else:
#             right=Vtree.internal_node(left=left, right=right)
#         left=Vtree.leaf_node(var=i+1)

#     return Vtree.internal_node(left=left, right=right)


vtree_format="""
c ids of vtree nodes start at 0
c ids of variables start at 1
c vtree nodes appear bottom-up, children before parents
c
c file syntax:
c vtree number-of-nodes-in-vtree
c L id-of-leaf-vtree-node id-of-variable
c I id-of-internal-vtree-node id-of-left-child id-of-right-child
c
"""

def right_linear_vtree(nb_vars, filename):
    with open("{}.vtree".format(filename), "w") as f:
        f.write(vtree_format)
        f.write("vtree {}\n".format(nb_vars))
        f.write("L 0 {}\n".format(nb_vars))
        f.write("L 1 {}\n".format(nb_vars-1))
        f.write("I {} 1 0\n".format(nb_vars))
        for i in range(2,nb_vars):
            f.write("L {} {}\n".format(i, nb_vars-i))
            f.write("I {} {} {}\n".format(nb_vars+i-1, i, nb_vars+i-2))

right_linear_vtree(41, "pizza")

# def graph2circuit(G, T, F, pos, neg, root=None, internal_nodes=None):
#     if root is None:
#         sources = [v for v in G.nodes() if G.in_degree(v)==0]
#         root=sources[0]

#     if internal_nodes is None:
#         internal_nodes={}

#     types=nx.get_node_attributes(G, "type")
#     lits=nx.get_edge_attributes(G, "lits")

#     # print(root, types[root])

#     if types[root]=="t":
#         internal_nodes[root] = T
#         return T, internal_nodes

#     elif types[root]=="f":
#         internal_nodes[root] = F
#         return F, internal_nodes

#     else:
#         children=[]
#         for edge in G.out_edges(root):
#             # print(edge)
#             if len(edge)==2:
#                 edge=edge+(0,)
#             if edge[1] in internal_nodes:
#                 node=internal_nodes[edge[1]]
#             else:
#                 node, internal_nodes = graph2circuit(G, T, F, pos, neg, root=edge[1], internal_nodes=internal_nodes)

#             if types[root]=="o" and len(lits[edge])>0:
#                 literals=[]
#                 for lit in lits[edge]:
#                     if int(lit)>0:
#                         literals.append(pos[int(lit)])
#                     else:
#                         literals.append(neg[abs(int(lit))])
#                 literals.append(node)
#                 children.append(dDNNF(type=Node.Types.AND, children=literals))

#             else:
#                 children.append(node)

#         if len(children)==1:
#             print("Only one children at : ", root)

#         if types[root]=="o":
#             node = dDNNF(type=Node.Types.OR, children=children)
#             internal_nodes[root] = node
#             return node, internal_nodes

#         elif types[root]=="a":
#             node = dDNNF(type=Node.Types.AND, children=children)
#             internal_nodes[root] = node
#             return node, internal_nodes
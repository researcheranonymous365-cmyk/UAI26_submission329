from enum import Enum

import warnings
import torch
import networkx as nx

from circuits.utils import memo, caching, logsumexp, logsum, log1mexp, keep_topk, smooth_topk

def topological_sort_edges(G):
    if not(isinstance(G, nx.DiGraph)):
        raise Exception("Only implemented for networkx DiGraph !")
    if not(nx.is_directed_acyclic_graph(G)):
        raise Exception("Only implemented for Directed Acyclic Graphs !")

    order = []
    for v in nx.topological_sort(G):
        for e in G.out_edges(v):
            order.append(e)

    return order

def init_literals(variables):
    litnodes={}
    for v in variables:
        litnodes[v] = Node(type=Node.Types.VAR, var=v)
        litnodes[-v] = Node.Neg(child=litnodes[v])
    
    return litnodes

def find_root(edgelist):
    for i in edgelist:
        root=True
        for j in edgelist:
            if i in edgelist[j]:
                root=False
        if root:
            return i
        
    raise Exception("The circuit should have a root node")

def iter_edgelist(edgelist, root=None):
    if root is None:
        root = find_root(edgelist)
    elif root in edgelist.keys():
        for node in edgelist[root].keys():
            iter_edgelist(edgelist, root=node)
        yield (root, edgelist[root])
    else:
        yield (root, None)


class Node():
    '''
    A node of a boolean circuit.
    '''
    class Types(Enum):
        F, T, VAR, NEG, AND, OR = range(1, 7)

    def __init__(self, type, children=None, id=None, var=None, manager=None, **kwargs):
        '''
        Initialize a node of a boolean circuit from its type and its children.
        The node is also identified with the boolean circuits of which it is the root.
        '''
        if not(type in Node.Types):
            raise Exception("The type of the node must be in : ", list(Node.Types))
        self.type  = type

        if children is None:
            children = []
        else:
            for c in children:
                if not(isinstance(c, Node)):
                    raise Exception("Children nodes must be of Node type")
                
        self.cache = {} # contains dynamic output of the nodes that change depending on the input (ex: mpe, pqe, etc.)
        self.cache["bit"] = False
        self.attributes = kwargs # contains static attributes of the node (ex: size, hieght, is_DNNF, etc.)
        # self.unique = None
        # self.vtree = None

        if self.is_binary() and (len(children)<=1):
            raise Exception("AND/OR nodes must have at least two children")
        elif (self.is_neg()) and (len(children)!=1):
            raise Exception("NEG nodes must have exactly one children")
        elif (self.is_leaf()) and (len(children)!=0):
            raise Exception("FALSE/TRUE/VAR nodes must have no children")
        else:
            self.children=children

        self.id = id
        self.var = var

        if (self.is_var()) and (var is None):
            raise Exception("LEAF nodes must have a variable")

        self.manager = manager
        if manager:
            manager.add(self, root=True)

    # SHORTCUT CONSTRUCTERS
    @classmethod
    def T(cls):
        return Node(type=Node.Types.T)
    
    @classmethod
    def F(cls):
        return Node(type=Node.Types.F)
    
    @classmethod
    def Var(cls, var):
        return Node(type=Node.Types.VAR, var=var)
    
    @classmethod
    def Neg(cls, child):
        return Node(type=Node.Types.NEG, children=[child])

    @classmethod
    def Or(cls, children, manager=None):
        if manager:
            node = manager.add(type=Node.Types.OR, children=children)
        else:
            node= Node(type=Node.Types.OR, children=children)
        return node
    
    @classmethod
    def And(cls, children, manager=None):
        if manager:
            node = manager.add(type=Node.Types.AND, children=children)
        else:
            node= Node(type=Node.Types.AND, children=children)
        return node
    
    # CONSTRUCTORS FROM FILES
    @classmethod
    def read_dimacs(cls, filename, manager=None):
        '''
        Initialize a CNF from a dimacs file.
        '''
        with open(filename) as f:
            lines = f.readlines()
            infos = lines[0]
            _, _, num_vars, num_clauses = infos.split()

            lit_nodes = []
            for v in range(1, int(num_vars)+1):
                lit=Node(type=Node.Types.VAR, var=v)
                lit_nodes.append(lit)
                lit_nodes.append(Node.Neg(child=lit))

            clauses= []
            for line in lines[1:]:
                lits = line.split()
                children = []
                for lit in lits[:-1]:
                    if lit[0]=="-":
                        v = int(lit[1:])
                        children.append(lit_nodes[2*v-1])
                    else:
                        v = int(lit)
                        children.append(lit_nodes[2*(v-1)])
                clauses.append(Node.Or(children=children))
        
        return Node.And(children=clauses)
    
    @classmethod
    def read_file(cls, filepath, litnodes=None):
        '''
        Initialize a boolean circuit from a nnf format file.
        '''
        with open(filepath) as f:
            lines = f.readlines()
            L = len(lines)+1
            variables = set()
            nodetypes = {}
            edgelits = {}
            for line in lines:
                e = line.split()
                if e[0].isnumeric():
                    lits = e[2:-1]
                    if e[0] in edgelits:
                        edgelits[e[0]][e[1]] = lits
                    else:
                        edgelits[e[0]] = {e[1]:lits}
                    variables.union(set(lits))
                else:
                    nodetypes[e[1]]=e[0]

        T = Node.T()
        F = Node.F()

        if litnodes is None:
            litnodes = init_literals(variables)
        root=find_root(edgelits)
        nodes={}

        for (n, edges) in iter_edgelist(edgelits, root):
            if edges is None:
                if nodetypes[n]=="t":
                    nodes[n]=T
                elif nodetypes[n]=="f":
                    nodes[n]=F
                else:
                    raise Exception("A node without edges should be a True or a False.")
                
            else:
                children = []
                for (c,lits) in edges.items():
                    children.append(Node.And(children=[litnodes[i] for i in lits].append(nodes[c])))

                if nodetypes[n]=="o":
                    nodes[n] = Node.Or(children=children)
                elif nodetypes[n]=="a":
                    nodes[n] = Node.And(children=children)

        return nodes[root]
    
    @classmethod
    def read_nnf(cls, filepath):
        '''
        Initialize a boolean circuit from a sdd format file.
        '''
        with open(filepath) as f:
            lines = f.readlines()

        lines = [l for l in lines if not l.startswith('c')]
        nnf_infos, lines = lines[0], lines[1:]
        nb_nodes = nnf_infos.split(" ")[1]
        # print(f"Building a sdd circuit with {nb_nodes} sdd nodes")

        nodes={}

        for l in lines:
            details = l.split(" ")
            t, i = details[0], int(details[-2])
            # print(f"Building node {i}")
            if t=="F":
                nodes[i] = Node.F()
            elif t=="T":
                nodes[i] = Node.T()
            elif t=="L":
                l = int(details[1])
                if l > 0:
                    nodes[i] = Node.Var(l-1)
                elif l < 0:
                    nodes[i] = Node.Neg(child=Node.Var(-l-1))
            elif t=="A":
                noe = int(details[1])
                elements = [int(details[i+2]) for i in range(noe)]
                nodes[i] = Node.And([nodes[e] for e in elements])
            elif t=="O":
                noe = int(details[1])
                elements = [int(details[i+2]) for i in range(noe)]
                nodes[i] = Node.Or([nodes[e] for e in elements])

        return nodes[0]

    @classmethod
    def term(cls, literals, litnodes=None):
        '''
        Initialize a boolean circuit representing the term corresponding to the given evidence.

        Inputs:
            evidence: a list of positive and negative integers that represent positive and negative literals respectively.
            litnodes: a list of existing literal nodes in the

        Outputs:
            node: the node equivalent to the conjunction of those literals.

        '''
        if litnodes is None:
            pos = {abs(int(lit)):Node(type=Node.Types.VAR, var=abs(int(lit))) for lit in literals}
            neg = {-lit:Node.Neg(child=n) for (lit, n) in pos.items()}
            litnodes = {**pos, **neg}

        node = Node.And(children=[litnodes[i] for i in literals])
        return node
    
    @classmethod
    def sfactor(cls, v:int, manager=None):
        '''
        Initialize a boolean circuit representing a smoothing factor corresponding to variable v.

        Inputs:
            v: an integer representing a variable
            manager: a node manager

        Outputs:
            node: the node equivalent to V | ~V
        '''
        nv = Node.Var(v)
        return Node.Or([nv, ~nv], manager=manager)
    
    # COMPILERS
    @classmethod
    def AcyclicSimplePath(cls, D):
        '''
        Compiles acyclic simple path constraints on an acyclic graph to a dDNNF circuit.

        Inputs:
            -D : a directed acyclic graph in networkx
        Outputs:
            -node : a root node of a dDNNF boolean circuit that represents simple path constraints on D.
        '''
        if not(isinstance(D, nx.DiGraph)):
            raise Exception("Only implemented for networkx DiGraph !")
        if not(nx.is_directed_acyclic_graph(D)):
            raise Exception("Only implemented for Directed Acyclic Graphs !")

        G = D.copy()

        sources = [v for v in G.nodes() if G.in_degree(v)==0]
        s=sources[0]
        targets = [v for v in G.nodes() if G.out_degree(v)==0]
        t=targets[0]
        

        # Merge all source vertices
        if len(sources)>1:
            toremove = []
            for v in sources[1:]:
                outedges = G.out_edges(v)
                G.add_edges_from([(s, e[1]) for e in outedges])
                # G.remove_edges_from(outedges)
                toremove += outedges
            G.remove_edges_from(toremove)
            G.remove_nodes_from(sources[1:])

        # Merge all target vertices
        if len(targets)>1:
            toremove = []
            for v in targets[1:]:
                inedges = G.in_edges(v)
                G.add_edges_from([(e[0], t) for e in inedges])
                toremove += inedges
            G.remove_edges_from(toremove)
            G.remove_nodes_from(targets[1:])

        V = list(nx.topological_sort(G))
        E = list(topological_sort_edges(G))

        nodes = {i:{} for i in range(len(E))}

        T = Node.T()
        F = Node.F()

        for (i,e) in enumerate(E):
            pos=Node.Var(i)
            neg=Node.Neg(child=pos)
            for (j,v) in enumerate(V):
                if i==0:
                    if v==s:
                        nodes[0][v]=neg
                    elif e==(s, v):
                        nodes[0][v]=pos
                    else:
                        nodes[0][v]=F

                else:
                    if e[1]==v: # if the target vertex of edge e is vertex v
                        if nodes[i-1][e[0]]==F and nodes[i-1][v]==F:
                            nodes[i][v]=F
                        else:
                            left=Node.And(children=[pos, nodes[i-1][e[0]]])
                            right=Node.And(children=[neg, nodes[i-1][v]])
                            nodes[i][v]=Node.Or(children=[left, right])
                    else:
                        if nodes[i-1][v]==F:
                            nodes[i][v]=F
                        else:
                            left=Node.And(children=[pos, F])
                            right=Node.And(children=[neg, nodes[i-1][v]])
                            nodes[i][v]=Node.Or(children=[left, right])

        root = nodes[len(E)-1][V[-1]]
        root.set_dDNNF()

        return root
    
    @classmethod
    def repr(type, cids=None, var=None, nnf=False, is_lit=False):
        '''
        Represents the node as a string.
        '''
        if type is Node.Types.F:
            st = 'F'
        elif type is Node.Types.T:
            st = 'T'
        elif type is Node.Types.VAR:
            if var is None:
                raise Exception("VAR needs var to be represented.")
            if nnf:
                st = 'L %d' % var+1
            else:
                st = 'V %d' % var+1
        elif type is Node.Types.NEG:
            if nnf:
                if not(is_lit):
                    raise Exception("NEG in nnf can only be literals.")
                elif var is None:
                    raise Exception("NEG need var to be represented in nnf ")
                else:
                    st = 'L %d' % var-1

            elif cids is None:
                raise Exception("NEG need cids to be represented not in nnf")
            else:
                st = 'N %d' % cids[0]
        elif type is Node.Types.OR:
            if cids is None:
                raise Exception("OR need cids to be represented not in nnf")
            st = 'O %d %s' % (len(cids), " ".join( '%d' % cid for cid in sorted(cids)))
        elif type is Node.Types.AND:
            if cids is None:
                raise Exception("AND need cids to be represented not in nnf")
            st = 'A %d %s' % (len(cids), " ".join( '%d' % cid for cid in sorted(cids)))
        
        return st


    # UNARY OPERATORS
    def __neg__(self):
        return Node.Neg(child=self)
    
    def __invert__(self):
        return Node.Neg(child=self)

    # BINARY OPERATORS
    def __and__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return Node.And(children=[self, other])
    
    def __or__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return Node.Or(children=[self, other])


    # CHECK TYPES
    def is_true(self):
        '''
        Check if the type of the node is True.
        '''
        if self.type is Node.Types.T:
            return True
        else:
            return False
    def is_false(self):
        '''
        Check if the type of the node is False.
        '''
        if self.type is Node.Types.F:
            return True
        else:
            return False
    def is_var(self):
        '''
        Check if the type of the node is a Variable.
        '''
        if self.type is Node.Types.VAR:
            return True
        else:
            return False
    def is_neg(self):
        '''
        Check if the type of the node is a Negation.
        '''
        if self.type is Node.Types.NEG:
            return True
        else:
            return False
    def is_or(self):
        '''
        Check if the type of the node is a Or.
        '''
        if self.type is Node.Types.OR:
            return True
        else:
            return False
    def is_and(self):
        '''
        Check if the type of the node is a And.
        '''
        if self.type is Node.Types.AND:
            return True
        else:
            return False

    @memo
    def is_leaf(self):
        '''
        Check if the node is a leaf of a boolean circuit (i.e. a variable or a boolean).
        '''
        if (self.is_true()) or (self.is_false()) or (self.is_var()):
            return True
        else:
            return False

    @memo
    def is_binary(self):
        '''
        Check if the node is a binary operator.
        '''
        if (self.is_or()) or (self.is_and()):
            return True
        else:
            return False

    @memo
    def is_lit(self):
        '''
        Check if the node is a literal (i.e. a variable or its negation).
        '''
        if (self.is_var()):
            return True
        elif (self.is_neg()):
            if self.children[0].is_var():
                return True
        else:
            return False

    @memo
    def is_decision(self):
        '''
        Check if the node is a decision node (i.e. an And node with the first child being a literal).
        '''
        if not(self.is_and()):
            return False
        elif (len(self.children) == 2) and self.children[0].is_lit():
            return True
        else:
            return False
    
    @memo
    def is_leftneg(self):
        '''
        Check if the node is a decision node (i.e. an And node with the first child being a literal).
        '''
        if not(self.is_and()):
            return False
        elif len(self.children) != 2:
            return False
        elif self.children[0].is_neg():
            if self.children[0].children[0].is_var():
                return True
        return False

    # CHECK CLASSES
    @memo
    def is_clause(self):
        '''
        Check if the node is a clause (i.e. a disjunction of literals).
        '''
        if self.is_lit():
            return True
        elif self.is_or():
            for c in self.children:
                if not(c.is_clause()):
                    return False
                return True
        else:
            return False

    @memo
    def is_term(self):
        '''
        Check if the node is a clause (i.e. a conjunction of literals).
        '''
        if self.is_lit():
            return True
        elif self.is_and():
            for c in self.children:
                if not(c.is_term()):
                    return False
                return True
        else:
            return False

    @memo
    def is_NNF(self):
        '''
        Check if the boolean circuit is in Negation Normal Form.
        '''
        if self.is_leaf() or self.is_lit():
            return True
        elif self.is_binary():
            for c in self.children:
                if not(c.is_NNF()):
                    return False
            return True
        else:
            return False

    @memo
    def is_smooth(self):
        '''
        Check if the boolean circuit is smooth.
        '''
        if not(self.is_or()):
            return True
        else:
            for c in self.children:
                if set(c.vars())!=set(self.vars()):
                    return False
            return True

    @memo
    def is_DNNF(self):
        '''
        Check if the boolean circuit is in Decomposable Negation Normal Form.
        '''
        if self.is_lit():
            return True
        elif not(self.is_NNF()):
            # print(self.string())
            return False
        else: # remains types AND/OR in NNF
            visited=[]
            for c in self.children:
                if not(c.is_DNNF()):
                    # print(self.string())
                    return False
                if (self.is_and()):
                    for v in c.vars():
                        if v in visited:
                            # print(self.string())
                            return False
                        visited.append(v)

            return True

    @memo
    def is_OBDD(self, order=None):
        '''
        Check if the boolean circuit is an Ordered Binary Decision Diagram.
        '''
        if order is None:
            order=[]
            for n in self.iter():
                if n.type is Node.Types.VAR and not(n.var in order):
                    order.append(n.var)
        
        v = order[0]
        rest = order[1:]

        if self.is_leaf():
            return True
        elif self.is_var():
            if self.var==v:
                return True
            else:
                return False
        elif self.is_neg():
            if not(self.children[0].is_var()):
                return False
            elif self.children[0].var==v:
                return True
            else:
                return False
        elif self.is_and():
            return False
        elif not(self.is_NNF()):
            return False
        elif len(self.children) != 2:
            return False
        else: # remains types OR in NNF with two children
            left, right = self.children[0], self.children[1]
            if not(left.is_decision()) or not(right.is_decision()):
                return False
            elif not(left.children[0].is_var()) or not(left.children[0].var==v):
                return False
            elif not(right.children[0].is_neg()) or not(right.children[0].children[0].is_var()) or not(right.children[0].children[0].var==v):
                return False
            else:
                if not(set(left.children[1].vars()).issubset(set(rest))):
                    return False
                elif not(set(right.children[1].vars()).issubset(set(rest))):
                    return False
                elif not(left.children[1].is_OBDD(rest)):
                    return False
                elif not(right.children[1].is_OBDD(rest)):
                    return False
                else:
                    return True
                
    # SET CLASS OF CIRCUITS
    def set_dDNNF(self, value=True):
        self.attributes.update({"is_dDNNF":value})
        for c in self.children:
            c.set_dDNNF(value)
                
    # ITERATE THROUGH NODES
    def iter(self,first_call=True, post=True):
        '''
        Generator of nodes, post or pre order
        '''
        if not(self.cache["bit"]):
            self.cache["bit"] = True

            if not(post):
                yield self

            if len(self.children)>0:
                for c in self.children:
                    for n in c.iter(first_call=False, post=post): yield n
            
            if post:
                yield self

        if first_call:
            self.clear_cache(key="bit", iterate=False)

    def clear_cache(self, key, iterate=False):
        if iterate:
            for node in self.iter():
                node.cache[key]=None
        elif key in self.cache and not(self.cache[key] is None):
            self.cache[key]=None
            if self.children:
                for c in self.children:
                    c.clear_cache(key, iterate=False)

    # Attributes
    @memo
    def vars(self):
        '''
        List the variables in the boolean circuit.
        '''
        if (self.is_var()):
            return [self.var]
        elif (self.is_false()) or (self.is_true()):
            return []
        else:
            _vars=[]
            for c in self.children:
                for v in c.vars():
                    if not(v in _vars):
                        _vars.append(v)
        return _vars

    @memo
    def height(self):
        '''
        Computes the height of the boolean circuit.
        '''
        if self.is_leaf():
            return 0
        else:
            return max([c.height() for c in self.children])+1

    @memo
    def size(self, first_call=True):
        '''
        Computes the size of the boolean circuit.
        '''
        if first_call:
            self.clear_cache("size")

        if self.is_leaf():
            return 0
        else:
            return sum([c.size(first_call=False)+1 for c in self.children])

    @memo
    def string(self):
        '''
        Represents the boolean circuit as a string.
        '''
        if self.is_true():
            return "T"
        elif self.is_false():
            return "F"
        elif self.is_var():
            return str(self.var)
        elif (self.is_neg()):
            return r"\neg ({})".format(self.children[0].string())
        elif (self.is_and()):
            return r"\land ({})".format(','.join(map(str,[c.string() for c in self.children])))
        elif (self.is_or()):
            return r"\lor ({})".format(','.join(map(str,[c.string() for c in self.children])))

    def set_ids(self, post=True, ids=None):
        '''
        Sets unique ids for each node in the boolean circuit.
        '''
        for (i, n) in enumerate(self.iter(post=post)):
            if ids:
                n.id=ids[i]
            else:
                n.id = i


    def repr(self, reset=False, nnf=False):
        '''
        Represents the node as a based on its type and the ids of its children or variable.
        '''
        if reset:
            self.set_ids()

        cids=[c.id for c in self.children] if not(self.is_leaf()) else None
        if self.is_var():
            var=self.var
        elif nnf and self.is_lit():
            var=-self.children[0].var
        else:
            var=None

        st = Node.repr(self.type, cids, var, nnf, self.is_lit())

        # if self.is_false():
        #     st = 'F'
        # elif self.is_true():
        #     st = 'T'
        # elif self.is_var():
        #     if nnf:
        #         st = 'L %d' % self.var+1
        #     else:
        #         st = 'V %d' % self.var+1
        # elif self.is_neg():
        #     if self.is_lit() and nnf:
        #         st = 'L %d' % (-(self.children[0].var+1))
        #     else:
        #         st = 'N %d' % self.children[0].id
        # elif self.is_or():
        #     st = 'O %d %s' % (len(self.children), " ".join( '%d' % c.id for c in self.children))
        # elif self.is_and():
        #     st = 'A %d %s' % (len(self.children), " ".join( '%d' % c.id for c in self.children))
        
        return st
    
    def write_nnf(self, filepath, ids=None, reset=False, header=None):
        '''
        Represents the circuit in nnf format
        '''
        if ids:
            self.set_ids(ids=ids)
        elif reset:
            self.set_ids()
        elif self.id is None:
            warnings.warn("Forcing ids reset because no id for the root node")
            self.set_ids()

        lines=['%s %d\n' % (n.repr(nnf=True), n.id) for n in self.iter()]

        with open(filepath,'w') as f:
            f.write(header)
            f.write('nnf %d \n' % len(lines))
            for l in lines:
                f.write(l)
        
        
    def primal_graph(self):
        '''
        Computes the primal graph of the boolean circuit in a networkx object.
        '''
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(self.vars())
        if self.is_and():
            for c in self.squeeze().children:
                edges_to_add = [(u, v) for u in c.vars() for v in c.vars() if u != v]
                G.add_edges_from(edges_to_add)

        else:
            edges_to_add = [(u, v) for u in self.vars() for v in self.vars() if u != v]
            G.add_edges_from(edges_to_add)

        return G

    # TRANSFORMATIONS
    @memo
    def to_negNNF(self):
        '''
        Returns the negation of the circuit in Negation Normal Form (NNF).
        '''
        if self.is_true():
            return Node.F()
        elif self.is_false():
            return Node.T()
        elif self.is_var():
            return Node.Neg(child=self)
        elif (self.is_neg()):
            return self.children[0]
        elif (self.is_and()):
            return Node.Or(children=[c.to_negNNF() for c in self.children])
        elif (self.is_or()):
            return Node.And(children=[c.to_negNNF() for c in self.children])
    
    @memo
    def to_NNF(self):
        '''
        Transforms the boolean circuit to a Negation Normal Form.
        '''
        if self.is_NNF():
            return self
        elif (self.is_neg()):
            return self.children[0].to_negNNF()
        elif (self.is_and()):
            return Node.And(children=[c.to_NNF() for c in self.children])
        elif (self.is_or()):
            return Node.Or(children=[c.to_NNF() for c in self.children])


    @memo
    def condition(self, evidence, first_call=True):
        """
        Conditions the circuit on evidence.

        Inputs:
            - evidence : a dictionary that maps each variable to either 1, 0, -1 if it is true, unspecified or false respectively.
        
        Outputs :
            - node : a circuit node that is equivalent to self | evidence.
        """
        
        if self.is_true() or self.is_false():
            return self
        elif self.is_var():
            if evidence[self.var]==1:
                return Node.T()
            elif evidence[self.var]==-1:
                return Node.F()
            else:
                return self
        elif self.is_lit() and self.is_neg():
            var=self.children[0].var
            if evidence[var]==1:
                return Node.F()
            elif evidence[var]==-1:
                return Node.T()
        else:
            new_children = [c.condition(evidence, first_call=False) for c in self.children]
            return Node(type=self.type, children=new_children)

        if first_call:
            self.clear_cache("condition")

    def distribute_evidence(self, e):
        """
        Distribute evidence by setting the value of the cache in all nodes of the circuits.

        Inputs:
            - e : a (batch_size x nb_vars) tensor that represents evidence with 1, 0, -1 values
        """

        if torch.all(e==0):
            self.cache["evidence"]=e
            if not(self.is_leaf()):
                for c in self.children:
                    c.distribute_evidence(e)

        else:
            if self.is_true:
                self.cache["evidence"]=e
            elif self.is_false:
                self.cache["evidence"]=torch.zeros_like(e)
            elif self.is_neg():
                self.cache["evidence"]=e
                self.children[0].distribute_evidence(torch.zeros_like(e))
            elif self.is_and():
                mask = torch.ones_like(e)
                mask[:, self.vars()]=0
                self.cache["evidence"]=torch.mul(mask, e)
                for c in self.children:
                    mask = torch.zeros_like(e)
                    mask[:, c.vars()]=1
                    c.distribute_evidence(torch.mul(mask, e))
            elif self.is_or():
                self.cache["evidence"]=torch.zeros_like(e)
                for c in self.children:
                    c.distribute_evidence(e)

    @memo
    def smoothed(self):
        if self.is_leaf() or self.is_lit():
            return self
        elif self.is_neg():
            return ~self.children[0].smoothed()
        elif self.is_and():
            return Node.And([c.smoothed() for c in self.children])
        else:
            children=[]
            for c in self.children:
                if set(c.vars())!=set(self.vars()):
                    diff=set(self.vars()).difference(set(c.vars()))
                    if len(diff)>=2:
                        sfactor = Node.And([Node.sfactor(v) for v in sorted(list(diff))], manager=self.manager)
                    else:
                        sfactor = Node.sfactor(list(diff)[0])
                    children.append(Node.And([c.smoothed(), sfactor]))
                else:
                    children.append(c.smoothed())
            return Node.Or(children)
        
    def cardinals(self, vars):
        '''
        Builds several circuits conditionned on the number of active variables in the models.

        Inputs:
            - vars : the set of variables on which to apply cardinal conditioning
        
        Outputs :
            - nodes : a set of nodes corresponding to self conditioned by the cardinality of variables in vars
        '''
        
        cmax = len(vars)

        for node in self.iter():
            if node.is_true():
                cardnodes = [self.T() for i in range(cmax)]
            elif node.is_false():
                cardnodes = [self.F() for i in range(cmax)]
            elif node.is_var():
                cardnodes = [self.F() for i in range(cmax)]
                if self.var in vars:
                    cardnodes[1] = self.T()
                else:
                    cardnodes[0] = self.T()
                
            elif node.is_neg():
                cardnodes = [self.F() for i in range(cmax)]
                cardnodes[0] = self.T()
            
            elif node.is_and():
                cnodes0, cnodes1 = node.children[0].cache["cardinals"], node.children[1].cache["cardinals"]
                cardnodes = []
                for i in range(cmax):
                    cardnodes.append(Node.Or([Node.And([cnodes0[j], cnodes1[i-j]]) for j in range(i)]))

            elif node.is_or():
                cnodes0, cnodes1 = node.children[0].cache["cardinals"], node.children[1].cache["cardinals"]
                cardnodes = [Node.Or([cnodes0[i], cnodes1[i]]) for i in range(cmax)]

            node.cache["cardinals"]=cardnodes

        cardnodes = self.cache["cardinals"]
        self.clear_cache("cardinals")
        return cardnodes


    # CONSISTENCY QUERIES

    @memo
    def accepts(self, states):
        """
        Computes if the circuit accepts a state or not.

        Inputs:
            - states : a (batch_size x num_variables) torch tensor of states of shape 
        
        Outputs : a (batch_size x 1) torch tensor that tells for each state if the circuit accepts it
        """

        if self.is_true():
            return torch.ones_like(states, dtype=torch.bool)
        elif self.is_false():
            return torch.zeros_like(states, dtype=torch.bool)
        elif self.is_var():
            return states[:,self.var]
        elif self.is_neg():
                return torch.logical_not(self.children[0].accepts(states))
        elif self.is_and():
            return torch.all(torch.stack([c.accepts(states) for c in self.children], dim=1), dim=1)
        elif self.is_or():
            return torch.any(torch.stack([c.accepts(states) for c in self.children], dim=1), dim=1)

    def co(self, evidence):
        '''
        Decides consistency of the circuit conditioned on the evidence.

        Inputs:
            - evidence : a torch tensor of partial assignements of shape (nb_evidence, num_variables)
        
        Outputs :
            - states : a tensor of shape (nb_evidence, num_variables) that represents satisfying states whenever the circuit is consistent with evidence
            - co : a tensor of shape (nb_evidence) that tells if the circuit is consistent with the evidence or not
        '''
        if not(self.is_DNNF()):
            raise Exception("The circuit must be in DNNF to decide its consistency")

        if not(evidence is None):
            es, n=evidence.shape
            if n != len(self.vars()):
                raise Exception("The dimension of evidence on the 1-axis must be the same as the number of variables in the circuit: expected {}, found {}".format(len(self.vars()), n))
            
        else:
            es = 1
            n = len(self.vars())
            evidence = torch.zeros(1,n)

        emask = torch.eq(evidence, 0) # compute the mask of evidence
        
        for node in self.iter():
            if node.is_true():
                co=torch.ones(es, dtype=torch.bool)
                states=evidence
            elif node.is_false():
                co=torch.zeros(es, dtype=torch.bool)
                states=evidence
            elif node.is_var():
                co=~torch.eq(evidence[:,node.var], -1)
                states = evidence
                states[:,node.var]=1
                
            elif node.is_neg():
                v=node.children[0].var
                co=~torch.eq(evidence[:,v], -1)
                states = evidence
                states[:,v]=-1
            
            elif node.is_and():
                statesc = torch.stack([c.cache["co"][0] for c in node.children], dim=2)
                coc = torch.stack([c.cache["co"][1] for c in node.children], dim=1)
                co = torch.all(coc, dim=1)
                states = torch.sum(statesc, dim=2) # since children nodes have distinct variables their mpe states so far are also disjoint

            elif node.is_or():
                mask=torch.zeros(es, n, dtype=torch.bool)
                mask[:,node.vars()]=True
                statesc = torch.stack([c.cache["co"][0] for c in node.children], dim=2)
                coc = torch.stack([c.cache["co"][1] for c in node.children], dim=1)
                co, idx = torch.max(coc, dim=1)
                idx=idx.unsqueeze(1).unsqueeze(2).expand(-1, statesc.shape[1], 1)
                states = torch.gather(statesc, dim=2, index=idx).squeeze(2)

            node.cache["co"]=(states, co)

        states, co = self.cache["co"]
        states = states.ge(0)
        self.clear_cache("co")
        return states, co

    # OPTIMIZATION QUERIES

    # def mpe(self, probs, evidence=None):
    #     '''
    #     Computes the most probable state given independent probabilities on the variables.

    #     Inputs:
    #         - probs : a torch tensor of probabilities of shape (batch_size, num_variables)
    #         - evidence : a torch tensor of partial assignements of shape (nb_evidence, num_variables)
        
    #     Outputs :
    #         - states : a torch tensor of shape (batch_size x nb_evidence, num_variables) representing for the most probable state
    #         - p : a torch tensor of shape (batch_size x nb_evidence) representing the probablity of the most probable state
    #     '''

    #     self.smoothed().clear_cache("mpe")
    #     bs, n=probs.shape

    #     if n != len(self.vars()):
    #         raise Exception("The dimension of probs on the 1-axis must be the same as the number of variables in the circuit: expected {}, found {}".format(len(self.vars()), n))

    #     if not(evidence is None):
    #         es = evidence.shape[0] # batch size of evidence
    #         if evidence.shape[1] != len(self.vars()):
    #             raise Exception("The dimension of evidence on the 1-axis must be the same as the number of variables in the circuit: expected {}, found {}".format(len(self.vars()), evidence.shape[1]))

    #     else:
    #         es = 1
    #         evidence = torch.zeros(1,n)

    #     default=torch.where(probs.ge(0.5), 1, -1).repeat(es,1)
    #     literals=torch.where(probs.ge(0.5), probs, 1-probs).repeat(es,1)

    #     evidence = evidence.repeat_interleave(bs, dim=0)
    #     emask = torch.eq(evidence, 0) # compute the mask of evidence

    #     eprobs = torch.prod(torch.where(emask, 1, torch.where(torch.eq(evidence, 1), probs.repeat(es,1), 1-probs.repeat(es,1))), dim=1)
        
    #     for node in self.smoothed().iter():
    #         if node.is_true():
    #             p=torch.ones(bs*es, dtype=torch.float64)
    #             states=evidence
    #         elif node.is_false():
    #             p=torch.zeros(bs*es, dtype=torch.float64)
    #             states=evidence
    #         elif node.is_var():
    #             vindex = torch.zeros_like(emask, dtype=torch.bool)
    #             vindex[:,node.var] = True
    #             states = torch.where(vindex, torch.where(emask, 1, evidence), 0)
    #             p = torch.where(emask[:,node.var],
    #                             probs[:,node.var].repeat(es),
    #                             torch.eq(evidence[:,node.var], 1).to(torch.float64))
                
    #         elif node.is_neg():
    #             statesc, pc = node.children[0].cache["mpe"]
    #             states, p = torch.where(emask, statesc.mul(-1), statesc), 1-pc
            
    #         elif node.is_and():
    #             statesc = torch.stack([c.cache["mpe"][0] for c in node.children], dim=2)
    #             pc = torch.stack([c.cache["mpe"][1] for c in node.children], dim=1)

    #             p = torch.prod(pc, dim=1)
    #             states = torch.sum(statesc, dim=2) # since children nodes have distinct variables their mpe states so far are also disjoint

    #         elif node.is_or():
    #             # mask=torch.zeros(bs*es, n, dtype=torch.bool)
    #             # mask[:,node.vars()]=True
    #             statesc = torch.stack([c.cache["mpe"][0] for c in node.children], dim=2)
    #             pc = torch.stack([c.cache["mpe"][1] for c in node.children], dim=1)
    #             # pc = torch.stack([torch.mul(c.cache["mpe"][1], torch.prod(torch.where(torch.logical_and(mask, c.cache["mpe"][0]==0), literals, 1), dim=1)) for c in node.children if not(c.is_false())], dim=1)

    #             p, idx = torch.max(pc, dim=1)
    #             idx=idx.unsqueeze(1).unsqueeze(2).expand(-1, statesc.shape[1], 1)
    #             states = torch.gather(statesc, dim=2, index=idx).squeeze(2)

    #         node.cache["mpe"]=(states, p)

    #     states, p = self.smoothed().cache["mpe"]
    #     states = states.ge(0)
    #     p = torch.mul(p, eprobs)
    #     self.smoothed().clear_cache("mpe")
    #     return states, p

    def mpe(self, probs, evidence=None):
        '''
        Computes the most probable state given independent probabilities on the variables.

        Inputs:
            - probs : a torch tensor of probabilities of shape (batch_size, num_variables)
            - evidence : a torch tensor of partial assignements of shape (nb_evidence, num_variables)
        
        Outputs :
            - states : a torch tensor of shape (batch_size x nb_evidence, num_variables) representing for the most probable state
            - p : a torch tensor of shape (batch_size x nb_evidence) representing the probablity of the most probable state
        '''
        if not(self.is_DNNF()):
            raise Exception("The circuit must be in DNNF to solve MPE")
        
        bs, n=probs.shape

        if n != len(self.vars()):
            raise Exception("The dimension of probs on the 1-axis must be the same as the number of variables in the circuit: expected {}, found {}".format(len(self.vars()), n))

        if not(evidence is None):
            es = evidence.shape[0] # batch size of evidence
            if evidence.shape[1] != len(self.vars()):
                raise Exception("The dimension of evidence on the 1-axis must be the same as the number of variables in the circuit: expected {}, found {}".format(len(self.vars()), evidence.shape[1]))

        else:
            es = 1
            evidence = torch.zeros(1,n)

        default=torch.where(probs.ge(0.5), 1, -1).repeat(es,1)
        literals=torch.where(probs.ge(0.5), probs, 1-probs).repeat(es,1)

        evidence = evidence.repeat_interleave(bs, dim=0)
        emask = torch.eq(evidence, 0) # compute the mask of evidence

        eprobs = torch.prod(torch.where(emask, 1, torch.where(torch.eq(evidence, 1), probs.repeat(es,1), 1-probs.repeat(es,1))), dim=1)
        
        self.clear_cache(key="mpe", iterate=True)
        for node in self.iter():
            if node.is_true():
                p=torch.ones(bs*es, dtype=torch.float64)
                states=evidence
            elif node.is_false():
                p=torch.zeros(bs*es, dtype=torch.float64)
                states=evidence
            elif node.is_var():
                vindex = torch.zeros_like(emask, dtype=torch.bool)
                vindex[:,node.var] = True
                states = torch.where(vindex, torch.where(emask, 1, evidence), 0)
                p = torch.where(emask[:,node.var],
                                probs[:,node.var].repeat(es),
                                torch.eq(evidence[:,node.var], 1).to(torch.float64))
                
            elif node.is_neg():
                statesc, pc = node.children[0].cache["mpe"]
                states, p = torch.where(emask, statesc.mul(-1), statesc), 1-pc
            
            elif node.is_and():
                statesc = torch.stack([c.cache["mpe"][0] for c in node.children], dim=2)
                pc = torch.stack([c.cache["mpe"][1] for c in node.children], dim=1)

                p = torch.prod(pc, dim=1)
                states = torch.sum(statesc, dim=2) # since children nodes have distinct variables their mpe states so far are also disjoint

            elif node.is_or():
                mask=torch.zeros(bs*es, n, dtype=torch.bool)
                mask[:,node.vars()]=True
                statesc = torch.stack([torch.where(torch.logical_and(mask, c.cache["mpe"][0]==0), default, c.cache["mpe"][0]) for c in node.children if not(c.is_false())], dim=2)
                pc = torch.stack([torch.mul(c.cache["mpe"][1], torch.prod(torch.where(torch.logical_and(mask, c.cache["mpe"][0]==0), literals, 1), dim=1)) for c in node.children if not(c.is_false())], dim=1)

                p, idx = torch.max(pc, dim=1)
                idx=idx.unsqueeze(1).unsqueeze(2).expand(-1, statesc.shape[1], 1)
                states = torch.gather(statesc, dim=2, index=idx).squeeze(2)

            node.cache["mpe"]=(states, p)

        states, p = self.cache["mpe"]
        states = states.ge(0)
        p = torch.mul(p, eprobs)
        # self.clear_cache("mpe")
        return states, p

    def topk(self, probs, k=2):
        '''
        Computes the k most probable states given independent probabilities on the variables.

        Inputs:
            - probs : a torch tensor of probabilities of shape (batch_size x num_variables)
            - first_call : allows to 
        
        Outputs :
            - states : a torch tensor of shape (batch_size x k x num_variables) representing for the most probable state
            - p : a torch tensor of shape (batch_size x k) representing the probablity of the most probable state
        '''
        if k==1:
            return self.mpe(probs=probs)
        
        bs, n = probs.shape
        default=torch.where(probs.ge(0.5), 1, -1).unsqueeze(-1)
        literals=torch.where(probs.ge(0.5), probs, 1-probs).unsqueeze(-1)
        
        for node in self.smoothed().iter():
            if node.is_true():
                p=torch.ones((bs, 1), dtype=torch.float64)
                states=torch.zeros((bs, n, 1), dtype=torch.float64)
            elif node.is_false():
                p=torch.zeros((bs, 1), dtype=torch.float64)
                states=torch.zeros((bs, n, 1), dtype=torch.float64)
            elif node.is_var():
                states=torch.zeros((bs, n, 1), dtype=torch.float64)
                states[:,node.var,:] = 1
                p=probs[:,node.var].unsqueeze(-1)
                
            elif node.is_neg():
                statesc, pc = node.children[0].cache["topk"]
                states, p = statesc.mul(-1), 1-pc
            
            elif node.is_and():
                statesn, pn = node.children[0].cache["topk"]
                for c in node.children[1:]:
                    statesc, pc = c.cache["topk"]
                    kn, kc = pn.shape[1], pc.shape[1]
                    pn = (pn[:, :, None] * pc[:, None, :]).reshape(bs, kn*kc)
                    statesn = (statesn[:, :, :, None] + statesc[:, :, None, :]).reshape(bs, n, kn*kc)
                    if kn*kc > k:
                        statesn, pn = keep_topk(statesn, pn, k=k, n=n)
                states, p = statesn, pn

            elif node.is_or():
                statesn = torch.cat([c.cache["topk"][0] for c in node.children], dim=2)
                pn = torch.cat([c.cache["topk"][1] for c in node.children], dim=1)
                
                if not(self.attributes.get("is_dDNNF", False)):
                    us = []
                    ps = []
                    for i in range(bs):
                        u, inverse = torch.unique(statesn[i], sorted=False, return_inverse=True, dim=1) # delete non-unique states in statesn
                        ind = torch.tensor([int((inverse == j).nonzero()[0]) for j in range(u.shape[1])])
                        p = pn[i,ind]
                        if u.shape[0] > k: # keep only top-k instances
                            u, p = keep_topk(u.unsqueeze(dim=0), p.unsqueeze(dim=0), k=k, n=n)
                        us.append(u)
                        ps.append(p)
                    
                    states = torch.cat(us, dim=0)
                    p = torch.cat(ps, dim=0)
                else:
                    states, p = keep_topk(statesn, pn, k=k, n=n)

            node.cache["topk"]=(states, p)

        states, p = self.smoothed().cache["topk"]
        states = states.ge(0)
        self.clear_cache("topk")
        return states, p
    
    # COUNTING QUERIES

    def mc(self, evidence):
        """
        Performs model counting.

        Inputs:
            - evidence : a torch tensor of partial assignements of shape (nb_evidence, num_variables)
        
        Outputs :
            - mc : a torch tensor of shape (nb_evidence) representing the number of satisfying assignements coherent with evidence
        """
        bs=evidence.shape[0]

        for node in self.iter():
            if node.is_true():
                mc = torch.zeros(bs)
            elif node.is_false():
                mc = torch.zeros(bs)
            elif node.is_var():
                mc = torch.where(torch.eq(evidence[:,node.var], -1), 0, 1)
            elif node.is_neg():
                mc = torch.where(torch.eq(evidence[:,node.children[0].var], 1), 0, 1)
            elif node.is_and():
                mc = torch.prod(torch.stack([c.cache["mc"] for c in node.children], dim=1), dim=1)
            elif node.is_or():
                nv = len(node.vars())
                sizes = torch.tensor([len(c.vars()) for c in node.children])
                smoothing = torch.pow(nv-sizes, 2)
                mc = torch.sum(torch.stack([smoothing[i]*c.cache["mc"] for (i,c) in enumerate(node.children)], dim=1), dim=1)

            node.cache["mc"]=mc

        mc=self.cache["mc"]
        self.clear_cache("mc")
        return mc
    
    def log_pqe(self, logprobs):
        #TODO:Add evidence
        """
        Performs PQE in log space.

        Inputs:
            - logprobs : a torch tensor of logprobs of shape (batch_size x num_variables)
        
        Outputs :
            - logp : a torch tensor of shape (batch_size) representing for the log-probability of the circuit under the independent multi-label distribution parameterized by probs.
        """
        bs=logprobs.shape[0]

        for node in self.iter():
            if node.is_true():
                logp = torch.zeros(bs, dtype=torch.float64)
            elif node.is_false():
                logp = torch.full((bs,), -300, dtype=torch.float64)
            elif node.is_var():
                logp = logprobs[:,node.var]
            elif node.is_neg():
                logp = log1mexp(node.children[0].cache["log_pqe"])
            elif node.is_and():
                logp = torch.sum(torch.stack([c.cache["log_pqe"] for c in node.children], dim=1), dim=1)
            elif node.is_or():
                logp = logsumexp(torch.stack([c.cache["log_pqe"] for c in node.children], dim=1), dim=1)

            node.cache["log_pqe"]=logp

        logp=self.cache["log_pqe"]
        self.clear_cache("log_pqe")

        return logp
        

    @memo
    def pqe(self, probs):
        #TODO:Add evidence
        """
        Performs PQE.

        Inputs:
            - probs : a torch tensor of probabilities of shape (batch_size x num_variables)
        
        Outputs :
            - p : a torch tensor of shape (batch_size) representing for the probability of the circuit under the independent multi-label distribution parameterized by probs.
        """
        if not(self.attributes.get("is_dDNNF", False)):
            raise Exception("Only implemented for dDNNFs !")
        
        bs=probs.shape[0]

        for node in self.iter():

            if node.is_true():
                pqe = torch.ones(bs)
            elif node.is_false():
                pqe = torch.zeros(bs)
            elif node.is_var():
                pqe = probs[:,node.var]
            elif node.is_neg():
                pqe = 1 - node.children[0].cache["pqe"]
            elif node.is_and():
                pqe = torch.prod(torch.stack([c.cache["pqe"] for c in node.children], dim=1), dim=1)
            elif node.is_or():
                pqe = torch.sum(torch.stack([c.cache["pqe"] for c in node.children], dim=1), dim=1)

            node.cache["pqe"]=pqe

        pqe=self.cache["pqe"]
        self.clear_cache("pqe")

        return pqe
    
    # FUZZY QUERIES
    @caching()
    def fuzzy(self, probs):
        '''
        Performs fuzzy evaluation.

        Inputs:
            - probs : a torch tensor of probabilities of shape (batch_size x num_variables)
        
        Outputs :
            - f : a torch tensor of shape (batch_size) representing for the fuzzy score of the circuit under probs.
        '''
        bs=probs.shape[0]

        if self.is_true():
            return torch.ones(bs)
        elif self.is_false():
            return torch.zeros(bs)
        elif self.is_var():
            return probs[:,self.var]
        elif self.is_neg():
                return 1 - self.children[0].fuzzy(probs)
        elif self.is_and():
                return torch.prod(torch.stack([c.fuzzy(probs) for c in self.children], dim=1), dim=1)
        elif self.is_or():
                return torch.sum(torch.stack([c.fuzzy(probs) for c in self.children], dim=1), dim=1)


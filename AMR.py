import re
from collections import OrderedDict
from nltk.tokenize import word_tokenize


"""
This script contains AMR processing functions and class definitions.
Use this script to read an AMR data file into sentence-AMR pairs.
Each pair is a tuple containing the english sentence split into a list of tokens
and its corresponding AMR graph.

Example usage (reading AMR data):
    import AMR
    from AMR import AMRNode, AMRGraph
    pairs = AMR.read_AMR_file('...')
    for sentence_tokens, AMR_graph in pairs:
        print('snt: ' + ' '.join(sentence_tokens))
        AMR_graph.print()
        
AMR is a rooted directed graph.
Each AMR node can have multiple children and parent nodes.
Only forward edges (children) are stored in this AMRNode representation.
An AMRNode is defined by its <inst> variable.
Each AMRNode contains a <child> object which is a dictionary of edges (keys) and child nodes (values)

Example usage (looping through AMR nodes starting from root):
    def process_node(amr_node):
        print(amr_node.inst)
        #procedure for this node
        for edge_name, child_node in node.child.items():
            if type(child_node) is str:
                #child_node is a literal value (eg: "Earth")
            elif type(child_node) is float:
                #child_node is a numeric value (eg: 2.0)
            elif type(child_node) is int:
                #child_node is a constant (eg: expressive)
            else:
                #child_node is an AMRNode
                process_node(amr_node)
                
    process_node(AMR_graph.root)

"""


"""Constant definitions
This dictionary of constants returns the index given a constant string

_constants = [None, '-', '+', 'imperative', 'interrogative', 'expressive']
constants = {}
for idx,_c in enumerate(_constants):
    constants[_c] = idx
"""    

def read_AMR_file(amr_file, acyclic=False):
    """Reads an AMR data file and returns all the sentence-amr pairs.
    Each pair is a tuple (sentence_tokens, AMR_graph).
    Set acyclic to True to return only acyclic AMR graphs
    """
    pairs = []
    snt = ''
    amr = ''
    i = 0
    
    with open(amr_file) as amr_read:
        for line in amr_read:
            i += 1
            if not line.strip():
                continue
            if '#' not in line:
                if snt:
                    amr += line
                else:
                    raise Exception("in line " + str(i) + "," + " Expecting: snt. Got: AMR")
            elif '::snt ' in line:
                if snt and amr:
                    tokens = word_tokenize(snt)
                    snt_tokens = [token.lower() for token in tokens]
                    amr_graph = AMRGraph(amr)
                    amr = ''
                    ind = line.find('::snt ') + 6
                    snt = line[ind:]
                    if acyclic:
                        if not amr_graph.is_cyclic():
                            pairs.append((snt_tokens, amr_graph))
                    else:
                        pairs.append((snt_tokens, amr_graph))
                elif snt:
                    raise Exception("in line " + str(i) + "," + " Expecting: AMR. Got: snt")
                else:
                    ind = line.find('::snt ') + 6
                    snt = line[ind:]
        if snt and amr:
            tokens = word_tokenize(snt)
            snt_tokens = [token.lower() for token in tokens]
            amr_graph = AMRGraph(amr)
            if acyclic:
                if not amr_graph.is_cyclic():
                    pairs.append((snt_tokens, amr_graph))
            else:
                pairs.append((snt_tokens, amr_graph))

    return pairs


def read_AMR_files(fnames, acyclic=False):
    pairs = []
    for fname in fnames:
        pairs += read_AMR_file(fname, acyclic)
    return pairs


"""Class definition for an AMR Node
Child is a dictionary object where keys are edges and values are the corresponding
AMR child nodes or literal values.
"""
class AMRNode:
    def __init__(self, instance, identity):
        self.inst = instance
        self.id = identity
        self.child = DupDict()
        self.parent = None
        self.numchild = 0
        
    def set_inst(self, instance):
        self.inst = instance
        
    def set_id(self, identity):
        self.id = identity
    
    def add_child(self, link, child):
        self.child[link] = child
        self.numchild += 1
    
    def remove_child(self, link):
        del self.child[link]
        self.numchild -= 1
    
    def add_literal(self, link, value):
        self.child[link] = value
    
    def add_numeric(self, link, value):
        self.child[link] = value
        

"""Class definition for an AMR Graph
Stores a dictionary of all AMR Nodes and the root Node
"""
class AMRGraph:   
    def __init__(self, string):
        self.nodes = None
        self.root = None
        self.size = 0
        AMRGraph.parse(self, string)
        
    def print(self):
        AMRGraph.print_node(self.root, '', [])
        
    def is_cyclic(self):
        return AMRGraph._cyclic(self.root, {})
    
    @staticmethod       
    def parse_node(node, tokens, i, nodes, ref, size):
        """Reads a set of AMR tokens to parse an empty node.
        Returns the final position of tokens after parsing the node completely.
        Adds unresolved references to the ref dictionary.
        """
        i += 1
        node_id = tokens[i]
        if node_id in nodes:
            newid = node_id + '0'
            while newid in nodes:
                newid += '0'
            node_id = newid
        node.set_id(node_id)
        nodes[node_id] = node
        i += 1
        if tokens[i] == '/':
            node.set_inst(tokens[i+1])
            i += 2
        
        #add children
        while tokens[i] != ')':
            if tokens[i] == ':':
                link = tokens[i+1]
                i += 2
                
                #Numeric
                fl = None
                try:
                    fl = float(tokens[i])
                except ValueError:
                    "Do nothing"
                if fl is not None:
                    node.add_numeric(link, fl)
                    size[0] += 1
                #Literal
                elif tokens[i][0] == '\"':
                    node.add_literal(link, tokens[i])
                    size[0] += 1
                #new node
                elif tokens[i] == '(':
                    new_node = AMRNode(None, None)
                    new_node.parent = node.id
                    node.add_child(link, new_node)
                    i = AMRGraph.parse_node(new_node, tokens, i, nodes, ref, size)
                #Reference or a constant
                else:
                    if node_id not in ref:
                        ref[node_id] = [(link, tokens[i])]
                    else:
                        ref[node_id].append((link, tokens[i]))
            
            i += 1
        
        return i
    
    @staticmethod
    def resolve_ref(nodes, ref, size):
        """Resolve all references to the AMR Nodes
        """
        for node_id in ref:
            for link,ref_node in ref[node_id]:
                #constants
                if ref_node not in nodes:
                    nodes[node_id].add_literal(link, ref_node)
                    size[0] += 1
                else:
                    nodes[node_id].add_child(link, nodes[ref_node])
    
    @staticmethod
    def parse(graph, string):
        """Fills an empty AMRGraph object from the given AMR string
        """
        tokens = re.findall(r'(\(|\)|\"[^\"]+\"|:|/|[^\s\(\):/\"]+)', string)
        if tokens[0] != '(':
            print('No root')
            return      
        graph.root = AMRNode(None, None)
        graph.nodes = OrderedDict()
        ref = {}
        size = [0]
        #parse root node
        AMRGraph.parse_node(graph.root, tokens, 0, graph.nodes, ref, size)
        #Resolve references
        AMRGraph.resolve_ref(graph.nodes, ref, size)
        
        graph.size = size[0] + len(graph.nodes)
        
    @staticmethod 
    def print_node(node, i, printed, end=' '):
        printed.append(node.id)
        print(i + '(' + node.id + ' / ' + node.inst, end=end)
        i += '  '
        for link,child_node in node.child.items():
            if type(child_node) is str or type(child_node) is float:
                print(i + ':' + link + ' ' + str(child_node), end=end)
            elif child_node.id in printed:
                print(i + ':' + link + ' ' + child_node.id, end=end)
            else:
                print(i + ':' + link, end=end)
                AMRGraph.print_node(child_node, i, printed)
        print(i[0:len(i)-1] + ')', end=end)
    
    @staticmethod
    def _cyclic(node, processing):
        if node in processing:
            return True
        processing[node] = True
        for link,child_node in node.child.items():
            if type(child_node) is str or type(child_node) is float:
                continue
            else:
                if AMRGraph._cyclic(child_node, processing):
                    return True
        if node in processing:
            del processing[node]
        return False


"""Class definition for a Dictionary that handles duplicate keys.
Each key can store multiple values.
This is useful because a Node in an AMR Graph can have multiple edges with the same name.
"""
class DupDict:
    def __init__(self):
        self.hash = {}
    
    def __getitem__(self, key):
        return self.hash[key]
    
    def __setitem__(self, key, value):
        if key in self.hash:
            self.hash[key].append(value)
        else:
            self.hash[key] = [value]
    
    def __delitem__(self, item):
        if type(item) is tuple:
            key,val = item
            l = self.hash[key]
            l.remove(val)
            if not l:
                del self.hash[key]
        else:
            del self.hash[key]
    
    def __contains__(self, key):
        return key in self.hash
    
    def __iter__(self):
        return self.hash.keys()
    
    def items(self):
        ditems = self.hash.items()
        tuples = []
        for key,values in ditems:
            for value in values:
                tuples.append((key, value))
        return tuples
    
    def values(self):
        return sum(self.hash.values(), [])



import re
from collections import OrderedDict
from nltk.tokenize import word_tokenize


"""Constant definitions
This dictionary of constants returns the index given a constant string
"""
_constants = [None, '-', '+', 'imperative', 'interrogative', 'expressive']
constants = {}
for idx,_c in enumerate(_constants):
    constants[_c] = idx
    

def read_AMR_file(amr_file):
    """Reads an AMR data file and returns all the sentence-amr pairs.
    Each pair is a tuple (sentence_tokens, AMR_graph).
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
                    pairs.append((snt_tokens, amr_graph))
                    amr = ''
                    ind = line.find('::snt ') + 6
                    snt = line[ind:]
                elif snt:
                    raise Exception("in line " + str(i) + "," + " Expecting: AMR. Got: snt")
                else:
                    ind = line.find('::snt ') + 6
                    snt = line[ind:]
        if snt and amr:
            tokens = word_tokenize(snt)
            snt_tokens = [token.lower() for token in tokens]
            amr_graph = AMRGraph(amr)
            pairs.append((snt_tokens, amr_graph))

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
    
    def add_constant(self, link, value):
        if value not in constants:
            self.child[link] = constants[None]
        else:
            self.child[link] = constants[value]
        

"""Class definition for an AMR Graph
Stores a dictionary of all AMR Nodes and the root Node
"""
class AMRGraph:   
    def __init__(self, string):
        self.nodes = None
        self.root = None
        AMRGraph.parse(self, string)
        
    def print(self):
        AMRGraph.print_node(self.root, '', [])
    
    @staticmethod       
    def parse_node(node, tokens, i, nodes, ref):
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
                #Literal
                elif tokens[i][0] == '\"':
                    node.add_literal(link, tokens[i])
                #new node
                elif tokens[i] == '(':
                    new_node = AMRNode(None, None)
                    new_node.parent = node.id
                    node.add_child(link, new_node)
                    i = AMRGraph.parse_node(new_node, tokens, i, nodes, ref)
                #Reference or a constant
                else:
                    if node_id not in ref:
                        ref[node_id] = [(link, tokens[i])]
                    else:
                        ref[node_id].append((link, tokens[i]))
            
            i += 1
        
        return i
    
    @staticmethod
    def resolve_ref(nodes, ref):
        """Resolve all references to the AMR Nodes
        """
        for node_id in ref:
            for link,ref_node in ref[node_id]:
                #constants
                if ref_node not in nodes:
                    nodes[node_id].add_constant(link, ref_node)                   
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
        #parse root node
        AMRGraph.parse_node(graph.root, tokens, 0, graph.nodes, ref)
        #Resolve references
        AMRGraph.resolve_ref(graph.nodes, ref)
        
    @staticmethod 
    def print_node(node, i, printed):
        printed.append(node.id)
        print(i + '(' + node.id + ' / ' + node.inst)
        i += '  '
        for link,child_node in node.child.items():
            if type(child_node) is str or type(child_node) is float:
                print(i + ':' + link + ' ' + str(child_node))
            elif child_node.id in printed:
                print(i + ':' + link + ' ' + child_node.id)
            else:
                print(i + ':' + link)
                AMRGraph.print_node(child_node, i, printed)
        print(i[0:len(i)-1] + ')')


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



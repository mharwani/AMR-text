import AMR

def process_node(encoder, amr_node, edge=None, context):
    print(amr_node.inst)
    #procedure for this node
    node_hidden = encoder.initHidden()
    for edge_name, child_node in amr_node.child.items():
        if type(child_node) in [str, float, int]:
            if type(child_node) is str:
                #child_node = word2idx(child_node)
        	op, hidden, edge_embed = encoder(child_node, encoder.initHidden(), edge, type(child_node))
            context = torch.cat([context, op])
            hidden.mul_(edge_embed)
            node_hidden.add_(hidden)
            #child_node is a literal value (eg: "Earth")
        else:
        	child_hidden, child_context = process_node(encoder, child_node, edge, context)
            node_hidden.add_(child_hidden)
            context = torch.cat([context, child_context])
            #child_node is an AMRNode
    op, hidden, edge_embed = encoder(amr_node.inst, node_hidden, edge)
    hidden.mul_(edge_embed)
    context = torch.cat([context, op])
    return hidden, context

def get_dataset(file_path):
    pairs = AMR.read_AMR_file(file_path)
    return pairs

dataset = get_dataset()
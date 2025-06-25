cpdef tuple build_cooccurrence_edges(dict word_to_node, list sentence_list, int max_length=-1):
    cdef:
        dict edge_counts = {}
        list sentence_nodes, filtered_nodes
        int i, j, node1, node2, node_idx
        object sentence, word
    
    for sentence in sentence_list:
        if sentence._word_objects is None:
            continue
        
        sentence_nodes = []
        for word in sentence._word_objects:
            node_idx = word_to_node.get(word.content, -1)
            if node_idx != -1:
                sentence_nodes.append(node_idx)
        if (max_length < 1):
            filtered_nodes = list(set(sentence_nodes))
        else: 
            filtered_nodes = list(set(sentence_nodes[:max_length]))
        
        for i in range(len(filtered_nodes)):
            for j in range(i + 1, len(filtered_nodes)):
                node1, node2 = filtered_nodes[i], filtered_nodes[j]
                if node1 > node2:
                    node1, node2 = node2, node1
                edge_counts[(node1, node2)] = edge_counts.get((node1, node2), 0) + 1
    
    return list(edge_counts.keys()), list(edge_counts.values())
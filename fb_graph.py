import numpy as np

from fb_model import FBModel
from model_builder import ModelBuilder
from collections import Counter
class FakeArgs:
    def __init__(self):
        self.img_size = 32
        self.num_dup_layers = 5
        self.in_channels = 1
        self.num_classes = 5
        self.max_depth = 6
        self.metaworld = False
        self.not_shorter_depth = True

args = FakeArgs()

class Node:
    def __init__(self, node_type, count):
        self.node_type = node_type
        self.count = count
        self.edges = []
    
    def add_edge(self, node):
        self.edges.append(node)

num_types = 6

fb_counts = np.zeros(num_types)
fb_probs = np.zeros(num_types)

fb_model = FBModel(args)

builder = ModelBuilder(args)
root_node = Node(-1, 1)
first_fb_node = Node(0, args.num_dup_layers)
assert builder.attempt_add_fb(fb_model._fbs[0], 0)
q = [(first_fb_node, [0], builder)]

# cur_node = root_node
# q = [(builder, root_node)]
while q:
    # Get current
    cur_node, cur_actions, cur_builder = q.pop(0)

    # Add next nodes/edges
    for i in range(num_types):
        # print("CUR BUILDER")
        if cur_builder.is_valid_fb(fb_model._fbs[i], len(cur_actions)):
             counts = dict(Counter(cur_actions))
             node_count = 0
             if i in counts:
                 node_count = counts[i]
             
             
             if args.num_dup_layers - node_count > 0:
                next_node = Node(i, args.num_dup_layers - node_count)
                cur_node.add_edge(next_node)
                next_builder = ModelBuilder(args)
                next_builder.cur_out_size = cur_builder.cur_out_size
                next_builder._prev_layer = cur_builder._prev_layer
                # print("NEXT BUILDER")
                next_builder.is_valid_fb(fb_model._fbs[i], len(cur_actions))
                assert next_builder.attempt_add_fb(fb_model._fbs[i], len(cur_actions))
                q.append((next_node, cur_actions + [i], next_builder))

def print_tree(node, cur_nodes):
    if not node:
        return
    
    # print(prev)
    if not node.edges:
        for n in cur_nodes:
            print(n.node_type, n.count)
        print("---")
    else:
        for e in node.edges:
            print_tree(e, cur_nodes + [e])
    
print_tree(first_fb_node, [first_fb_node])


for i in range(128):
    cur_node = first_fb_node
    fb_probs[cur_node.node_type] += 1/cur_node.count
    fb_counts[cur_node.node_type] += 1
    for j in range(args.max_depth):
        total_count = 0
        for next_node in cur_node.edges:
            total_count += next_node.count
        
        for next_node in cur_node.edges:
            fb_probs[next_node.node_type] += 1/total_count
            fb_counts[next_node.node_type] += 1

        if cur_node.edges:
            cur_node = np.random.choice(cur_node.edges)
        else:
            break
print(fb_probs)
print(fb_counts)
#print("fb_probs")
print(fb_probs/fb_counts)





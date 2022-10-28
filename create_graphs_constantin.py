from pathlib import Path
from matplotlib import pyplot as plt
from treelib import Node, Tree
import networkx as nx
import json
import numpy as np


# Move this to a new module ?
def create_tree(conversation_id, conversation_data):
    """conversation_data should already be in the right chronological order"""
    tree = Tree()
    tree.create_node(conversation_id, conversation_id)  # Create Root Node

    error_count = 0
    og_reply_count = 0
    for tree_datapoint in conversation_data:
        try:
            tree.create_node(
                tree_datapoint[0][0], tree_datapoint[0][0], parent=tree_datapoint[1][0]
            )
            if tree_datapoint[0][0] == conversation_id:
                og_reply_count += 1
        except:
            error_count += 1

    print(
        f"Created Tree with {len(conversation_data) - error_count} replies. {error_count} replies were throwing "
        f"an error due to deletions."
    )
    width = len(list(tree.filter_nodes(lambda x: tree.depth(x) == 1)))

    return tree.depth(), tree.size(), width, og_reply_count


def create_graph(conv_id, conversation_data):
    """conversation_data should already be in the right chronological order"""
    G = nx.Graph()
    unique_users = [conversation_data[0][1][1]] #add OG user
    for graph_datapoint in conversation_data:
        G.add_edge(graph_datapoint[0][1], graph_datapoint[1][1])
        unique_users.append(graph_datapoint[0][1])

    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
    structural_virality = nx.wiener_index(G) / (
            G.number_of_nodes() * (G.number_of_nodes() - 1)
    )
    density = nx.classes.function.density(G)
    diameter = nx.diameter(G)
    unique_users = np.unique(unique_users)
    nx.draw(G, node_size=2)
    plt.savefig(f"plots/graph_users_{conv_id}.png")

    return structural_virality, density, diameter, len(unique_users)


if __name__ == "__main__":
    # For now only with dem tweets, when we have reps we can
    # simply do "+ json load(reps_file)" to concatenate the
    # lists and assemble all the trees at once
    # Load data from files
    tweets = json.load(open("data/dem_tweets_v2.json"))
    convos_edges = {}

    for tweet in tweets:
        try:
            convos_edges[tweet["conversation_id"]] = (
                json.load(
                    open(
                        f"data/conversations/conversation_with_authors_{tweet['conversation_id']}.json"
                    )
                ),
                "party",  # Need to get this data from existing files
            )
        except:
            print(
                f"Conversation data for tweet {tweet['conversation_id']} is not present"
            )

    conversation_features = []

    print("Loaded the conversations data in memory correctly.")

    Path("plots/").mkdir(exist_ok=True)

    # For each conversation, assemble the tree and return some metrics
    # Maybe here we should also print the plots? Although that might
    # take a lot of space. Maybe we could create a list of how many
    # nodes they have, then we sort it, and then we'll plot the 10
    # biggest democrats and 10 biggest republicans trees / graphs
    # For now it's pretty straightforward, we only compute the depth here
    for conv_idx, (conversation_id, (edges, party)) in enumerate(convos_edges.items()):
        if conv_idx == 100:
            break
        print(f"{conv_idx} out of {len(convos_edges)}")
        print(
            "Creating tree / graph for conversation",
            conversation_id,
            "that has",
            len(edges),
            "edges",
        )
        depth, size, width, og_reply_count = create_tree(conversation_id, edges[::-1])
        structural_virality, density, diameter, unique_users = create_graph(conversation_id, edges[::-1])
        conversation_features.append(
            {"conversation_id": conversation_id, "party": party, "depth": depth, "size": size, "width": width,
             "structural_virality": structural_virality, "density": density, "diameter": diameter,
             "reply_count": og_reply_count, "unique_users": unique_users}
        )

    # For now as json but maybe we could do .csv or pickle
    json.dump(conversation_features, open("conversation_metrics.json", "w"))


#Ideas: Investigate how often OG replies to tweets -> might be sign for democrat or Rep
#Maybe also investigate to how many people OG replies in graph
#Reciprocity
#Assortativity
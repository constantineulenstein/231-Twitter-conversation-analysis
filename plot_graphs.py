import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt


def create_graph(conv_id, is_directed=False):
    conversation_data = (
        json.load(open(f"data/conversations/conversation_with_authors_{conv_id}.json")),
    )[0]
    """conversation_data should already be in the right chronological order"""
    G0 = nx.DiGraph() if is_directed else nx.Graph()

    og_author = conversation_data[0][1][1]
    unique_users = [og_author]
    og_reply_count = 0

    for ((_, replier_id), (_, author_id)) in conversation_data:
        G0.add_edge(replier_id, author_id)
        unique_users.append(replier_id)
        if replier_id == og_author:
            og_reply_count += 1

    Gcc = sorted(
        nx.weakly_connected_components(G0)
        if G0.is_directed()
        else nx.connected_components(G0),
        key=len,
        reverse=True,
    )
    G = G0.subgraph(Gcc[0])

    plt.figure(figsize=(12, 12))
    nx.draw(G, node_size=3)
    plt.savefig(f"plots/graph_users_{conv_id}.png")
    plt.close()


if __name__ == "__main__":
    create_graph(1583843365233504257)

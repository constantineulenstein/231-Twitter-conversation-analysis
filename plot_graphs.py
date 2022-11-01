from pathlib import Path
import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

def create_graph(conv_id, title, is_directed=False):
    node_size = 10
    
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
    G: nx.Graph = G0.subgraph(Gcc[0])

    plt.figure(figsize=(12, 12))
    nx.draw(G, node_size=node_size)
    plt.savefig(f"plots_max_graphs/graph_{title}_nx{'_di' if is_directed else ''}.png")
    plt.close()
    
    pos = graphviz_layout(G, prog="dot")
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, node_size=node_size)
    plt.savefig( f"plots_max_graphs/graph_{title}_dot{'_di' if is_directed else ''}.png")
    plt.close()
    
    pos = graphviz_layout(G, prog="osage")
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, node_size=node_size)
    plt.savefig( f"plots_max_graphs/graph_{title}_osage{'_di' if is_directed else ''}.png")
    plt.close()
    
    pos = graphviz_layout(G, prog="fdp")
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, node_size=node_size)
    plt.savefig( f"plots_max_graphs/graph_{title}_fdp{'_di' if is_directed else ''}.png")
    plt.close()
    
    pos = graphviz_layout(G, prog="neato")
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, node_size=node_size)
    plt.savefig( f"plots_max_graphs/graph_{title}_neato{'_di' if is_directed else ''}.png")
    plt.close()
    
    pos = graphviz_layout(G, prog="twopi")
    plt.figure(figsize=(12, 12))
    nx.draw(G, pos, node_size=node_size)
    plt.savefig( f"plots_max_graphs/graph_{title}_twopi{'_di' if is_directed else ''}.png")
    plt.close()


if __name__ == "__main__":
    Path("plots_max_graphs/").mkdir(exist_ok=True)
    
    max_convs = json.load(open("maxConvs.json"))
    min_convs = json.load(open("minConvs.json"))
    
    for feature in max_convs:
        create_graph(max_convs[feature]["conversation_id"], f"{feature}_max", is_directed=True)
        create_graph(min_convs[feature]["conversation_id"], f"{feature}_min", is_directed=True)

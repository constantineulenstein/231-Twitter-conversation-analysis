import json
from treelib import Node, Tree
from tweet_stream import create_twitter_client
import networkx as nx
import matplotlib.pyplot as plt

def create_tree(conversation_id, conversation_data):
    """conversation_data should already be in the right chronological order"""
    tree = Tree()
    tree.create_node(conversation_id, conversation_id)  # Create Root Node

    error_count = 0
    for tree_datapoint in conversation_data:
        try:
            tree.create_node(
                tree_datapoint[0], tree_datapoint[0], parent=tree_datapoint[1]
            )
        except:
            error_count += 1
    tree.show()
    print(
        f"Created Tree with {len(conversation_data) - error_count} replies. {error_count} replies were throwing "
        f"an error due to deletions."
    )
    depth_1_nodes = list(tree.filter_nodes(lambda x: tree.depth(x) == 1))
    print(f"Tree size is {tree.size()}, tree depth is {tree.depth()} and tree width is {len(depth_1_nodes)}.")


def create_tree_with_networkx(conversation_id, conversation_data):
    """conversation_data should already be in the right chronological order"""
    G = nx.Graph()
    G.add_node(conversation_id)
    error_count = 0
    for tree_datapoint in conversation_data:
        try:
            G.add_edge(
                tree_datapoint[0], tree_datapoint[1]
            )
        except:
            error_count += 1
    print(
        f"Created Tree with {len(conversation_data) - error_count} replies. {error_count} replies were throwing "
        f"an error due to deletions."
    )

    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    print(f"structural virality is {nx.wiener_index(G0)/(G.number_of_nodes()*(G.number_of_nodes()-1))}")
    nx.draw(G0, node_size=3)
    plt.savefig("test.png")


def create_graph(conversation_id, conversation_data, client):
    """conversation_data should already be in the right chronological order"""
    G = nx.Graph()
    tweet_to_author_dict = {}

    og_author_id = client.get_tweet(conversation_id, expansions=["author_id"]).data.author_id
    tweet_to_author_dict[conversation_id] = og_author_id
    error_count = 0
    for graph_datapoint in conversation_data:
        try:
            if graph_datapoint[0] in tweet_to_author_dict:
                reply = tweet_to_author_dict[graph_datapoint[0]]
            else:
                reply = client.get_tweet(graph_datapoint[0], expansions=["author_id"]).data.author_id
            og = tweet_to_author_dict[graph_datapoint[1]]
                #if graph_datapoint[1] in tweet_to_author_dict \
                #else client.get_tweet(graph_datapoint[1], expansions=["author_id"]).data.author_id
            G.add_edge(reply, og)
            if graph_datapoint[0] not in tweet_to_author_dict:
                tweet_to_author_dict[graph_datapoint[0]] = reply
        except:
            error_count += 1
    print(
        f"Created Tree with {len(conversation_data) - error_count} replies. {error_count} replies were throwing "
        f"an error due to deletions."
    )
    print(G)


if __name__ == "__main__":
    twitterclient = create_twitter_client()

    tweets_f = open('data/dem_tweets_v2.json')
    tweets = json.load(tweets_f)
    example_id = tweets[116]['conversation_id']
    conv_f = open(f'data/conversations/conversation_{example_id}.json')
    conv = json.load(conv_f)
    create_tree_with_networkx(example_id, conv[::-1])
    #create_graph(example_id, conv[::-1], twitterclient)
    #create_tree(example_id, conv[::-1])

# Look at unique users -> number of nodes in graph
from pathlib import Path
from time import sleep
from matplotlib import pyplot as plt
from treelib import Node, Tree
import networkx as nx
import json
import numpy as np
import tweepy

from tweet_stream import create_twitter_client


def get_reply_with_reply_proportion(tree, cutoff):
    depth_1_nodes = list(tree.filter_nodes(lambda x: tree.depth(x) == 1))
    counter = 0
    if len(depth_1_nodes) == 0:
        return 0
    for depth_1_node in depth_1_nodes:
        subtree_len = len(Tree(tree.subtree(depth_1_node.identifier)).all_nodes())
        if subtree_len > cutoff:
            counter += 1
    return counter / len(depth_1_nodes)


# Move this to a new module ?
def create_tree(conversation_id, conversation_data):
    """conversation_data should already be in the right chronological order"""
    tree = Tree()
    tree.create_node(conversation_id, conversation_id)  # Create Root Node

    error_count = 0
    for tree_datapoint in conversation_data:
        try:
            tree.create_node(
                tree_datapoint[0][0], tree_datapoint[0][0], parent=tree_datapoint[1][0]
            )
        except:
            error_count += 1

    print(
        f"Created Tree with {len(conversation_data) - error_count} replies. {error_count} replies were throwing "
        f"an error due to deletions."
    )
    width = len(list(tree.filter_nodes(lambda x: tree.depth(x) == 1)))

    reply_to_reply_proportion = len(
        list(tree.filter_nodes(lambda x: tree.depth(x) > 1))
    ) / len(list(tree.all_nodes()))
    reply_with_reply_proportion = get_reply_with_reply_proportion(tree, 2)

    return (
        tree.depth(),
        tree.size(),
        width,
        reply_to_reply_proportion,
        reply_with_reply_proportion,
    )


def create_graph(conv_id, conversation_data, is_directed=False, should_plot=False):
    """conversation_data should already be in the right chronological order"""
    G = nx.DiGraph() if is_directed else nx.Graph()

    og_author = conversation_data[0][1][1]
    unique_users = [og_author]
    og_reply_count = 0

    for ((_, author_id), (_, replier_id)) in conversation_data:
        G.add_edge(author_id, replier_id)
        unique_users.append(author_id)
        if author_id == og_author:
            og_reply_count += 1

    Gcc = sorted(
        nx.weakly_connected_components(G)
        if G.is_directed()
        else nx.connected_components(G),
        key=len,
        reverse=True,
    )
    G = G.subgraph(Gcc[0])

    unique_users = np.unique(unique_users)
    density = nx.classes.function.density(G)

    if should_plot:
        nx.draw(G, node_size=2)
        plt.savefig(f"plots/graph_users_{conv_id}.png")
        plt.close()

    if not G.is_directed():
        trials_to_do = max(1000, 4 * len(conversation_data))
        average_clustering = nx.approximation.average_clustering(G, trials=trials_to_do)
        diameter = nx.approximation.diameter(G)
        return (
            average_clustering,
            density,
            diameter,
            len(unique_users),
            og_reply_count,
        )
    else:
        assortativity = nx.degree_assortativity_coefficient(G, x="in", y="out")
        reciprocity = nx.reciprocity(G)
        if np.isnan(assortativity):
            return 0
        return assortativity, reciprocity


def compile_graph_data(dem_tweets, rep_tweets):
    twitterclient = create_twitter_client()

    if Path("data/followers_counts.json").exists():
        print("Loading followers counts in memory")
        followers_counts = json.load(open("data/followers_counts.json"))
    else:
        followers_counts = {}
        for tweet in dem_tweets + rep_tweets:
            # We had forgotten to retrieve followers_count, so that had to be done here
            if tweet["author_id"] in followers_counts:
                continue
            try:
                followers_count = (
                    twitterclient.get_user(
                        id=tweet["author_id"], user_fields=["public_metrics"]
                    ).data["public_metrics"]["followers_count"],
                )
            except tweepy.TooManyRequests:
                print("Too Many Requests, I sleeps for 15mn")
                sleep(15 * 60)
                followers_count = (
                    twitterclient.get_user(
                        id=tweet["author_id"], user_fields=["public_metrics"]
                    ).data["public_metrics"]["followers_count"],
                )

            print(
                f"Pulled for {tweet['full_name']} ({tweet['author_id']}): {followers_count} followers."
            )
            followers_counts[tweet["author_id"]] = followers_count
        json.dump(followers_counts, open("data/followers_counts.json", "w"))

    convos_edges = []

    def add_tweet_data(tweet, party):
        try:
            convo_data = (
                tweet["conversation_id"],
                json.load(
                    open(
                        f"data/conversations/conversation_repulled_{tweet['conversation_id']}.json"
                    )
                ),
                party,
                followers_counts[str(tweet["author_id"])][0],
            )
            convos_edges.append(convo_data)
        except Exception as e:
            print(
                f"Conversation data for tweet {tweet['conversation_id']} is not present,"
                " maybe tweet was deleted and so no data pulled ?"
            )

    for tweet in dem_tweets:
        add_tweet_data(tweet, "Democrat")

    for tweet in rep_tweets:
        add_tweet_data(tweet, "Republican")

    return convos_edges


if __name__ == "__main__":
    output_file_name = "conversation_metrics_v6.json"
    # Set to true to observe if our computations of average_clustering are precise
    should_compute_approximation_metrics = False

    dem_tweets = json.load(open("data/dem_tweets_v2.json"))
    rep_tweets = json.load(open("data/rep_tweets_v2.json"))

    convos_edges = compile_graph_data(dem_tweets, rep_tweets)

    print("Loaded the conversations data in memory.")

    conversation_features = []
    Path("plots/").mkdir(exist_ok=True)

    convos_edges.sort(key=lambda c: len(c[1]))  # Sort by edges count

    for conv_idx, (conversation_id, edges, party, follower_count) in enumerate(
        convos_edges[::-1]
    ):
        print(f"{conv_idx} out of {len(convos_edges)}")
        print(
            "Conversation",
            conversation_id,
            "that has",
            len(edges),
            "edges",
        )
        print("Computing repies tree metrics")
        (
            depth,
            size,
            width,
            reply_to_reply_proportion,
            reply_with_reply_proportion,
        ) = create_tree(conversation_id, edges[::-1])
        print("Computing undirected repliers graph metrics")
        (
            average_clustering,
            density,
            diameter,
            unique_users,
            og_reply_count,
        ) = create_graph(conversation_id, edges[::-1])
        print("Computing directed repliers graph metrics")
        assortativity, reciprocity = create_graph(conversation_id, edges[::-1], is_directed=True)

        if should_compute_approximation_metrics:
            clusterings = []
            for _ in range(10):
                (
                    average_clustering,
                    density,
                    diameter,
                    unique_users,
                    og_reply_count,
                ) = create_graph(conversation_id, edges[::-1])
                clusterings.append(average_clustering)

            print(
                np.mean(clusterings),
                np.std(clusterings),
                np.std(clusterings) / np.mean(clusterings),
            )
            average_clustering = np.mean(clusterings)

        conversation_features.append(
            {
                "conversation_id": conversation_id,
                "party": party,
                "depth": depth,
                "size": size,
                "width": width,
                "average_clustering": average_clustering,
                "density": density,
                "diameter": diameter,
                "assortativity": assortativity,
                "reciprocity": reciprocity,
                "reply_count": og_reply_count,
                "unique_users": unique_users,
                "reply_to_reply_proportion": reply_to_reply_proportion,
                "reply_with_reply_proportion": reply_with_reply_proportion,
                "follower_count": follower_count,
            }
        )

        if (conv_idx - 1) % 100 == 0:
            json.dump(conversation_features, open(output_file_name, "w"))

    json.dump(conversation_features, open(output_file_name, "w"))

# Ideas: Investigate how often OG replies to tweets -> might be sign for democrat or Rep
# Maybe also investigate to how many people OG replies in graph
# Reciprocity
# Assortativity

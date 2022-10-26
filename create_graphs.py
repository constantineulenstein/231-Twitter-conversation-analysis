from treelib import Node, Tree
import json

# Move this to a new module ?
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

    return tree.depth()


if __name__ == "__main__":
    # For now only with dem tweets, when we have reps we can
    # simply do "+ json load(reps_file)" to concatenate the
    # lists and assemble all the trees at once
    # Load data from files
    tweets = json.load(open("data/dem_tweets_v1.json"))
    convos_edges = {
        tweet_data["conversation_id"]: (
            open(
                f'data/conversations/conversation_{tweet_data["conversation_id"]}.json'
            ),
            "party",  # Need to get this data from existing files
        )
        for tweet_data in tweets
    }

    conversation_features = []

    print("Loaded the conversations data in memory correctly.")

    # For each conversation, assemble the tree and return some metrics
    # Maybe here we should also print the plots? Although that might
    # take a lot of space. Maybe we could create a list of how many
    # nodes they have, then we sort it, and then we'll plot the 10
    # biggest democrats and 10 biggest republicans trees / graphs
    # For now it's pretty straightforward, we only compute the depth here
    for conversation_id, (edges, party) in convos_edges.values():
        print(
            "Creating tree for conversation",
            conversation_id,
            "that has",
            len(edges),
            "edges",
        )
        depth = create_tree(conversation_id, edges)
        conversation_features.append(
            {"conversation_id": conversation_id, "party": party, "depth": depth}
        )

    # For now as json but maybe we could do .csv or pickle
    json.dump(conversation_features, open("conversation_metrics.json", "w"))
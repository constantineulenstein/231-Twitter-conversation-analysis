#!/usr/bin/env python3
"""Collect tweets from Twitter streaming API via tweepy"""
import sys
import time
import pandas as pd
import json
import datetime
from treelib import Node, Tree
from pathlib import Path

from tweepy import Stream, Client, StreamingClient, StreamRule, Paginator


def create_twitter_client():
    creds = {}
    for line in open("creds.txt", "r"):
        row = line.strip()
        if row:
            key, value = row.split()
            creds[key] = value

    return Client(
        creds["bearer_token"],
        creds["api_key"],
        creds["api_secret"],
        creds["token"],
        creds["token_secret"],
    )


def get_conversation_data(conv_id, client):
    request_args = {
        "query": f"conversation_id:{conv_id}",
        "expansions": ["referenced_tweets.id.author_id", "in_reply_to_user_id"],
        "tweet_fields": ["in_reply_to_user_id"],
    }
    conversation = client.search_recent_tweets(**request_args)

    conversation_data = []
    # list to store all conversation interactions -> later reverse it because twitter
    # api is in reversed chronological order

    for conv in conversation.data:
        conversation_data.append((conv["id"], conv["referenced_tweets"][0]["id"]))

    while "next_token" in conversation.meta:
        conversation = client.search_recent_tweets(
            **request_args,
            next_token=conversation.meta["next_token"],
        )

        for conv in conversation.data:
            conversation_data.append((conv["id"], conv["referenced_tweets"][0]["id"]))

    return conversation_data[::-1]


def retrieve_and_populate_tweets_data_for_user(
    client, user_id: int, official_name: str, next_token: str = None
):
    today = datetime.datetime.today()
    week_ago = today - datetime.timedelta(days=7)

    request_args = {
        "id": user_id,
        "expansions": ["referenced_tweets.id.author_id"],
        "start_time": week_ago,
        "tweet_fields": ["conversation_id,author_id"],
        "max_results": 100,
    }

    populated_tweets = []

    tweets = client.get_users_tweets(**request_args, pagination_token=next_token)

    for tweet in tweets.data:
        if tweet.referenced_tweets is None:
            tweet_data = {
                "full_name": official_name,
                "author_id": tweet["author_id"],
                "conversation_id": tweet["conversation_id"],
                "tweet_text": tweet["text"],
            }
            populated_tweets.append(tweet_data)

    return populated_tweets, tweets.meta


def get_legislator_tweets(client):
    legs = pd.read_pickle("legislators")
    democrat_tweets = []
    republican_tweets = []

    for idx in legs.index:
        try:
            user_id = int(legs["social.twitter_id"][idx])
        except ValueError:
            print("Twitter id not present for", official_name)
        official_name = legs["name.official_full"][idx]
        party = legs["party"][idx]

        print(
            "Total tweets pulled so far:",
            len(democrat_tweets),
            "democrats, and",
            len(republican_tweets),
            "republican",
        )
        print("Pulling tweets for legislator", official_name, "of the", party, "party")

        try:
            tweets, tweets_meta = retrieve_and_populate_tweets_data_for_user(
                client,
                user_id,
                official_name,
            )
        except TypeError:
            # For example, Lindsey Graham's account in the data we pulled is not the
            # one he uses, it's a very old one.
            print("No tweets found for", official_name, ", is the account correct ?")

        while "next_token" in tweets_meta:
            new_tweets, tweets_meta = retrieve_and_populate_tweets_data_for_user(
                client,
                user_id,
                official_name,
                tweets_meta["next_token"],
            )

            tweets += new_tweets

        if party == "Democrat":
            democrat_tweets += tweets
        else:
            republican_tweets += tweets

    return democrat_tweets, republican_tweets


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


if __name__ == "__main__":
    twitterclient = create_twitter_client()

    dem_tweets, rep_tweets = get_legislator_tweets(twitterclient)

    now = datetime.datetime.now().strftime("%Y_%m_%d_%Y_%I_%M%p")

    data_folder = Path("data/").mkdir(exist_ok=True)

    # Save this already collected data in json files
    json.dump(dem_tweets, open(f"data/dem_tweets_{now}.json ", "w"))
    json.dump(rep_tweets, open(f"data/rep_tweets_{now}.json ", "w"))

    # Get the convo data from twitter and write it in a file
    for tweet in dem_tweets:
        tweet["conversation_data"] = get_conversation_data(
            tweet["conversation_id"], twitterclient
        )
    for tweet in rep_tweets:
        tweet["conversation_data"] = get_conversation_data(
            tweet["conversation_id"], twitterclient
        )

    json.dump(dem_tweets, open(f"data/dem_tweets_with_convo_{now}.json ", "w"))
    json.dump(rep_tweets, open(f"data/rep_tweets_with_convo_{now}.json ", "w"))

    # create_tree(conversation_id, conversation_data)

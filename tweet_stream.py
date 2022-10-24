#!/usr/bin/env python3
"""Collect tweets from Twitter streaming API via tweepy"""
import sys
import time
import pandas as pd
import datetime
from treelib import Node, Tree

from tweepy import Stream, Client, StreamingClient, StreamRule, Paginator


def create_twitter_client():
    creds = {}
    for line in open("creds.txt", "r"):
        row = line.strip()
        if row:
            key, value = row.split()
            creds[key] = value

    return Client(
        creds["bearer_token"], creds["api_key"], creds["api_secret"], creds["token"], creds["token_secret"]
    )


def create_tree(conv_id, client):
    tree = Tree()
    conversation = client.search_recent_tweets(f"conversation_id:{conv_id}",
                                               expansions=['referenced_tweets.id.author_id',
                                                           "in_reply_to_user_id"],
                                               tweet_fields=["in_reply_to_user_id"])
    tree.create_node(conv_id, conv_id)  # Create Root Node

    conversation_tree_data = []  # list to store all conversation interactions -> later reverse it because twitter
    # api is in reversed chronological order

    for conv in conversation.data:
        conversation_tree_data.append((conv["id"], conv["referenced_tweets"][0]["id"]))

    while "next_token" in conversation.meta:
        conversation = client.search_recent_tweets(f"conversation_id:{conv_id}",
                                                   expansions=['referenced_tweets.id.author_id',
                                                               "in_reply_to_user_id"],
                                                   tweet_fields=["in_reply_to_user_id"],
                                                   next_token=conversation.meta["next_token"])

        for conv in conversation.data:
            conversation_tree_data.append((conv["id"], conv["referenced_tweets"][0]["id"]))

    error_count = 0
    for tree_datapoint in conversation_tree_data[::-1]:  # reverse list
        try:
            tree.create_node(tree_datapoint[0], tree_datapoint[0], parent=tree_datapoint[1])
        except:
            error_count += 1
    tree.show()
    print(f"Created Tree with {len(conversation_tree_data) - error_count} replies. {error_count} replies were throwing "
          f"an error due to deletions.")


def get_legislator_tweets(client):
    legs = pd.read_pickle("legislators")
    democrat_tweets = []
    republican_tweets = []

    today = datetime.datetime.today()
    week_ago = today - datetime.timedelta(days=7)

    for idx in legs.index:
        tweets = client.get_users_tweets(int(legs["social.twitter_id"][idx]),
                                         expansions=['referenced_tweets.id.author_id'], start_time=week_ago,
                                         tweet_fields=["conversation_id,author_id"])
        for tweet in tweets.data:
            if tweet.referenced_tweets is None:
                democrat_tweets.append(
                    [legs["name.official_full"][idx], tweet["author_id"], tweet["conversation_id"], tweet["text"]]) if \
                    legs["party"][idx] == "Democrat" else republican_tweets.append([tweet["id"], tweet["text"]])

        while "next_token" in tweets.meta:
            tweets = client.get_users_tweets(int(legs["social.twitter_id"][idx]),
                                             expansions=['referenced_tweets.id.author_id'], start_time=week_ago,
                                             tweet_fields=["conversation_id"],
                                             pagination_token=tweets.meta["next_token"])

            for tweet in tweets.data:
                if tweet.referenced_tweets is None:
                    democrat_tweets.append(
                        [legs["name.official_full"][idx], tweet["author_id"], tweet["conversation_id"],
                         tweet["text"]]) if \
                        legs["party"][idx] == "Democrat" else republican_tweets.append([tweet["id"], tweet["text"]])

        if idx == 1:  # This is here to only fetch last twenty tweets. If removed, it will fetch the last 7 days
            break

        return democrat_tweets, republican_tweets


twitterclient = create_twitter_client()

dem_tweets, rep_tweets = get_legislator_tweets(twitterclient)

conversation_id = dem_tweets[0][2]
create_tree(conversation_id, twitterclient)

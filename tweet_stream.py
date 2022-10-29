#!/usr/bin/env python3
"""Collect tweets from Twitter streaming API via tweepy"""
import time
import pandas as pd
import json
import datetime
from pathlib import Path
from glob import glob

from tweepy import Client, TooManyRequests


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
    Path("data/conversations").mkdir(exist_ok=True)

    # See if cached data exists and if so, return it
    cached_data_file = Path(
        f"data/conversations/conversation_with_authors_{conv_id}.json"
    )
    if cached_data_file.exists():
        cached_data = json.load(open(cached_data_file, "r"))
        print(
            f"Conversation",
            conv_id,
            "data exists already, using",
            len(cached_data),
            "cached tweets...",
        )
        return cached_data

    request_args = {
        "query": f"conversation_id:{conv_id}",
        "expansions": [
            "author_id",
            "referenced_tweets.id.author_id",
            "in_reply_to_user_id",
        ],
        "tweet_fields": ["author_id", "in_reply_to_user_id"],
        "max_results": 100,
    }
    try:
        conversation = client.search_recent_tweets(**request_args)
    except TooManyRequests:
        print("Too many requests sent recently, waiting 15mn")
        time.sleep(15 * 60)
        conversation = client.search_recent_tweets(**request_args)

    conversation_data = []
    # list to store all conversation interactions -> later reverse it because twitter
    # api is in reversed chronological order

    try:
        for conv in conversation.data:
            conversation_data.append(
                (
                    [conv["id"], conv["author_id"]],
                    [conv["referenced_tweets"][0]["id"], conv["in_reply_to_user_id"]],
                )
            )
    except TypeError:
        print(
            "Conversation",
            conv_id,
            "has no data, maybe no one answered or the tweet was deleted.",
        )
        return conversation_data

    while "next_token" in conversation.meta:
        try:
            conversation = client.search_recent_tweets(
                **request_args,
                next_token=conversation.meta["next_token"],
            )

            for conv in conversation.data:
                conversation_data.append(
                    (
                        [conv["id"], conv["author_id"]],
                        [
                            conv["referenced_tweets"][0]["id"],
                            conv["in_reply_to_user_id"],
                        ],
                    )
                )

        except TooManyRequests:
            print("Too many requests sent recently, waiting 15mn")
            time.sleep(15 * 60)

    # Write in the cached data
    json.dump(conversation_data, open(cached_data_file, "w"))

    print(
        f"Pulled",
        len(conversation_data),
        "tweets for conversation",
        conv_id,
    )

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
            continue

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
            continue

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


if __name__ == "__main__":
    # This code is necessary to populate data/ with tweets from republicans and
    # dems, and to get the associated conversations
    twitterclient = create_twitter_client()

    # dem/rep_tweets_v1.json is the data from Tue 18 4:30pm to Tue 25 4:30pm
    # Not in the repo right now bcoz it's too big to be with the code
    # Set this value to True if you want to re-pull tweets from the past 7 days
    use_cached_tweets = True
    if use_cached_tweets is False:
        dem_tweets, rep_tweets = get_legislator_tweets(twitterclient)

        now = datetime.datetime.now().strftime("%Y_%m_%d_%Y_%I_%M%p")

        data_folder = Path("data/").mkdir(exist_ok=True)

        # Cache this already collected data in json files
        json.dump(dem_tweets, open(f"data/dem_tweets_{now}.json", "w"))
        json.dump(rep_tweets, open(f"data/rep_tweets_{now}.json", "w"))
    else:
        dem_tweets = json.load(open("data/dem_tweets_v2.json", "r"))
        rep_tweets = json.load(open("data/rep_tweets_v2.json", "r"))

    # Get the convo data from twitter and write them in files
    for tweet_idx, tweet in enumerate(dem_tweets):
        print(
            "Getting conversation data for tweet",
            tweet_idx,
            "out of",
            len(dem_tweets) - 1,
            "tweets from democrats",
        )
        tweet["conversation_data"] = get_conversation_data(
            tweet["conversation_id"], twitterclient
        )
    for tweet_idx, tweet in enumerate(rep_tweets):
        print(
            "Getting conversation data for tweet",
            tweet_idx,
            "out of",
            len(rep_tweets) - 1,
            "tweets from republicans",
        )
        tweet["conversation_data"] = get_conversation_data(
            tweet["conversation_id"], twitterclient
        )

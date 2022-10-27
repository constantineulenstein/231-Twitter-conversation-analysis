###TIMESTAMP OF FILES: Wed 19 Oct 11:00 (PST)

import pandas as pd
from yaml import safe_load
import pickle

with open("legislators-current.yaml", "r") as f:
    legs_current = pd.json_normalize(safe_load(f))


party = pd.Series([row[-1]["party"] for row in legs_current["terms"]])

legs_current = legs_current[
    ["name.official_full", "id.bioguide", "id.thomas", "id.govtrack"]
]
legs_current["party"] = party
# legs_current = legs_current[legs_current["name.official_full"]=="Elizabeth Warren"]


with open("legislators-social-media.yaml", "r") as f:
    legs_social = pd.json_normalize(safe_load(f))

legs_social = legs_social[
    ["id.bioguide", "id.thomas", "id.govtrack", "social.twitter_id", "social.twitter"]
]

legs = pd.merge(
    legs_current,
    legs_social,
    how="inner",
    left_on=["id.bioguide", "id.thomas", "id.govtrack"],
    right_on=["id.bioguide", "id.thomas", "id.govtrack"],
)
legs = legs.drop(["id.bioguide", "id.thomas", "id.govtrack"], axis=1)
legs = legs[(legs.party == "Democrat") | (legs.party == "Republican")]


legs.to_pickle("legislators")


# Population 1: Democrats
# Population 2: Republicans
# Hypothesis: Republican trees go deeper, have wider spread, and larger cascade

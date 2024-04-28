import json
import pprint
import random

with open("playground/howto100m/caption.json") as hand:
    ht100m = json.load(hand)
    pprint.pprint(len(ht100m))

print(
    random.sample(list(ht100m.keys()), 50)
)


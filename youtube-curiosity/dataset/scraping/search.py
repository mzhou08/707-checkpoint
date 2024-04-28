import nltk
from nltk.corpus import wordnet as wn
import os
import random
import json
import multiprocessing
import tqdm
import time

import pytube

TAGS = ["IMG", "DSC", "MOV", "MVI"]

def get_wordnet_lemmas():
    nltk.download('wordnet')
    # all words
    return [w.replace("_", " ") for w in wn.words()]

def get_random_tag_search(tag, n_queries):
    # returns a list of n_queries strings where each is of the form
    # [tag]XXXX, where XXXX is a random number and tag is IMG, DSC, MOV, MVI etc.
    n_queries = min(n_queries, 10000)
    query_ints = random.sample(range(0, 10000), n_queries)

    return [f"{tag}{i}" for i in query_ints]

def search_youtube(query, num_results_pages=3):
    """
    Uses PyTube to search YouTube for a query, and return a list of resulting video IDs.
    """
    # a list of pytube.YouTube objects
    s = pytube.Search(query)
    # Adds an additional num_results_pages pages of results
    for _ in range(num_results_pages):
        try:
            s.get_next_results()
        except IndexError:
            break

    try:
        return [v.video_id for v in s.results]
    except KeyError:
        # some properties, such as CommandMetadata, cause a KeyError
        return []

def execute_queries(queries, output_file):
    """
    Executes a list of queries, and writes the results to a JSON file.
    """
    start = time.time()

    video_ids = set()

    with multiprocessing.Pool() as pool:
        max_ = len(queries)
        with tqdm.tqdm(total=max_) as pbar:
            for i, search_results in enumerate(pool.imap_unordered(search_youtube, queries)):
                video_ids |= set(search_results)
                pbar.update()

    end = time.time()
    n_channels = len(video_ids)

    print(f"retrieved {n_channels} channels in {end - start} seconds. {(end - start) / n_channels} seconds per channel")

    # write channel ids to json
    json.dump(
        list(video_ids),
        open(output_file, "w"),
    )

if __name__ == "__main__":
    BASE_FILE_PATH = "/grogu/user/mhzhou/youtube-curiosity/dataset/scraping/"
    
    wordnet_lemmas = get_wordnet_lemmas()
    tag_queries = []
    for tag in TAGS:
        tag_queries += get_random_tag_search(tag, 100)

    execute_queries(wordnet_lemmas, f"{BASE_FILE_PATH}wordnet_videos.json")
    execute_queries(tag_queries, f"{BASE_FILE_PATH}tag_videos.json")

import requests
import urllib
import numpy as np
import pandas as pd
import time
import os
import wget

# load list of team IDs and the corresponding collection names on NeuroVault
team_data = pd.read_csv("../data/team_level_data.csv")

# let's try a few files to start
team_data = team_data[0:2]

# specify which types of images we're interested in
files_to_get = ["hypo1_unthresh", "hypo2_unthresh"]

for i, (index, team) in enumerate(team_data.iterrows()):  # loop through teams
    print("Downloading data for team %d / %d" % (i + 1, len(team_data)))

    # build url to get the list of images that the given team uploaded
    collection_n = team["neurovault_collection_number"]
    url = f"https://neurovault.org/api/collections/{collection_n}/images"

    # get the response as a json (this contains the list of images)
    response = requests.get(url)
    json = response.json()

    # download relevant images
    for item in json["results"]:
        if item["name"] in files_to_get:
            file_url = item["file"]
            file_path = "../data/%s_%s.nii.gz" % (
                item["name"],
                team["team_id"],
            )
            print("Downloading file from ", file_url)

            # using requests
            # file_response = requests.get(file_url, timeout=10)
            # if not os.path.exists(file_path):
            #     with open(file_path, "wb") as f:
            #         f.write(response.content)
            #         f.flush()

            # using urllib
            urllib.request.urlretrieve(file_url, file_path)

            # using wget
            # wget.download(file_url)

            # pause between downloads to be nice to the API
            print("Pausing... \n")
            time.sleep(5)  # in sec

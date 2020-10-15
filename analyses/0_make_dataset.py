from nilearn.datasets import fetch_neurovault_ids
import requests
import pandas as pd
import IPython

# load list of team IDs and the corresponding collection names on NeuroVault
team_data = pd.read_csv("../data/team_level_data.csv")

# team_data = team_data[0:4]

# specify which images we're interested in, by image name
# we're only interested in the images for the 1st and 2nd hypotheses
images_to_get = ["hypo1_unthresh", "hypo2_unthresh"]

# specify the columns in which to store the image ids for each hypothesis
columns = ["neurovault_image_id_hypo1", "neurovault_image_id_hypo2"]

for i, (index, team) in enumerate(team_data.iterrows()):  # loop through teams
    print(
        "Getting metadata for team %d/%d" % (i + 1, len(team_data)), end="\r"
    )

    # build url to get the list of images that the given team uploaded
    collection_n = team["neurovault_collection_number"]
    url = f"https://neurovault.org/api/collections/{collection_n}/images"

    # get the response as a json
    # (this contains the list of images and corresponding metadata)
    response = requests.get(url)
    json = response.json()

    # get image ids for relevant images
    for item in json["results"]:
        for image_to_get, column in zip(images_to_get, columns):
            if item["name"] in images_to_get:
                image_id = int(item["id"])

                # sore image id in the row (indexed by 'index') of the
                # given collection id
                collection_id = int(item["collection_id"])
                index = team_data.index[
                    team_data.neurovault_collection_number == collection_id
                ]
                team_data.loc[index, column] = image_id

print(team_data.neurovault_image_id_hypo1)
# team_data.to_csv("../data/team_level_data.csv")
print("\n")

# specify which types of images we're interested in
image_ids_hypo1 = tuple(team_data.neurovault_image_id_hypo1.astype(int))
image_ids_hypo2 = tuple(team_data.neurovault_image_id_hypo2.astype(int))
image_id_list = [image_ids_hypo1, image_ids_hypo2]

# specify where to store the data directories
data_dir_hypo1 = "../data/hypo1"
data_dir_hypo2 = "../data/hypo2"
data_dir_list = [data_dir_hypo1, data_dir_hypo2]

# specify the columns where we'll store the paths to the image files
columns = ["image_path_hypo1", "image_path_hypo2"]

for image_ids, data_dir, column in zip(image_id_list, data_dir_list, columns):
    # Download the images
    print("Downloading data")
    bunch = fetch_neurovault_ids(
        # collection_ids=collection_ids,
        image_ids=image_ids,
        mode="download_new",
        data_dir=data_dir,
        fetch_neurosynth_words=False,
        vectorize_words=False,
        verbose=3,
    )

    # save the paths to the images in the DataFrame containing the
    # team-level information
    for image_path in bunch.images:
        # find the collection number in the path to make sure we add the path
        # to the correct row
        end = -len("/image_133243.nii.gz")
        start = end - 4
        collection_id = int(image_path[start:end])

        # get the index of the given collection
        index = team_data.index[
            team_data.neurovault_collection_number == collection_id
        ]

        # store the path
        team_data.loc[index, column] = image_path

    team_data.to_csv("../data/team_level_data.csv")
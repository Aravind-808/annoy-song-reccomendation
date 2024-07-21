from annoy import AnnoyIndex
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings(action='ignore',category=SyntaxWarning)

# Function to classify a genre into one of the categories
def classify_genre(genre):
    genre_categories = {
    'pop': ['pop', 'indie-pop', 'k-pop', 'cantopop', 'j-pop', 'latin', 'malay', 'mandopop', 'pop-film'],
    'rock': ['rock', 'alt-rock', 'punk-rock', 'grunge', 'garage', 'psych-rock', 'rock-n-roll', 'rockabilly'],
    'rap': ['hip-hop', 'rap'],
    'dance': ['electronic', 'techno', 'house', 'edm', 'trance', 'dubstep', 'deep-house', 'disco', 'synth-pop', 'electro', 'detroit-techno', 'minimal-techno', 'hardstyle'],
    'rnb': ['r-n-b', 'soul', 'gospel'],
    'folk': ['folk', 'acoustic','bluegrass', 'country', 'singer-songwriter', 'mpb', 'samba', 'tango', 'forro', 'pagode', 'flamenco'],
    'jazz': ['jazz', 'blues', 'swing'],
    'metal': ['metal', 'black-metal', 'death-metal', 'heavy-metal', 'metalcore', 'grindcore', 'punk'],
    'world': ['world-music', 'reggae', 'iranian', 'turkish', 'latino', 'spanish', 'indian'],
    'miscellaneous': ['ambient', 'anime', 'comedy', 'children', 'disney', 'goth', 'emo', 'industrial', 'show-tunes', 'happy']
    }
    for category, genres_list in genre_categories.items():
        if genre in genres_list:
            return category
    return 'Other'  # If genre doesn't match any category, classify as 'Other'

# Encoding The genre (giving it a numerical value) so that the model can load it
def encodegenre(dataframe):
    genre_df = pd.DataFrame(data= df_copy["track_genre"].unique(),index=df_copy["track_genre"].unique())
    label = LabelEncoder()
    df_copy["track_genre"] = label.fit_transform(df_copy["track_genre"])
    genre_df["encoded"] = df_copy["track_genre"].unique()
    genre_df.drop(0, axis=1, inplace=True)
    return genre_df

# Annoy implementation - Target vector initialized, then actual annoy code.
def annoynn(dataframe, target_vector_dataframe):
    target_vec_arr = target_vector_dataframe.values.flatten().tolist()
    if(len(target_vec_arr) > 11):
        target_vec_arr = target_vec_arr[:11]
    # print(target_vec_arr)

    if(len(target_vec_arr)==0):
        print("No Such Song found!!")
        exit(0)

    # the next few lines is just building the model :P
    index = AnnoyIndex(metric="angular", f = dataframe.shape[1])
    for i in range(len(dataframe)):
        vec = dataframe.iloc[i].values # Initializing values and giving it an index, because annoy needs an index (label) for each vector.
        index.add_item(i, vec) # Adding the vectors and indices to the AnnoyIndex model
    index.build(n_trees=200)

    # find 500 nearest neighbours of the target vector
    k=500

    # Finding indices of nearest neighbours
    nearest_neighbours = index.get_nns_by_vector(target_vec_arr, k)

    # Returning only results from specified genre
    nearest_neighbours = [idx for idx in nearest_neighbours if dataframe.iloc[idx]['track_genre'] == genrenum]
    return nearest_neighbours

# read the dataset
df = pd.read_csv("SongsCleaned.csv")
df.drop("dummy", axis=1, inplace=True)

# I used this code to clean the dataset. make sure to remove a '\' wherever there are two of those.
'''
df["artists"] = df["artists"].str.replace(';',', ').str.split(r'\\s*\\,\\s*',n=1).str[0]

# remove feature names from song because you cant memorize it all
# also convert the song names to lowercase because im too lazy to search for songs with case sensitivity
df["track_name"] = df["track_name"].str.split(r'\\s*\\(\\s*',n=1).str[0]
df["track_name"] = df["track_name"].str.lower()

duplicates = df.duplicated(subset= ["track_name", "artists"], keep= 'first')
df = df[~duplicates]
'''

# Apply the classification function to the 'genre' column, if you are using the online dataset and if you want to
# df['track_genre'] = df['track_genre'].apply(classify_genre)

# Check for stuff in the df
'''
print(df.columns)
print(df.isnull().sum())
print(df.duplicated().any())
'''

# drop unwanted stuff
df_copy = df.drop(["Unnamed: 0","track_id","mode","artists", "album_name","duration_ms","time_signature","key","loudness"], axis=1)
df_copy["explicit"] = df_copy["explicit"].astype(int)


# Encoding all the different genres in track_genre
genre_df = encodegenre(df_copy)

# I used this to check if the genre encoding worked
genre_df.to_csv("Genres.csv")

# get track name and genre as string input
trackname = input("Enter Song that you are listening to right now: ") 
genre = input("Enter Genre: ")
if(genre not in genre_df.index):
    print("No such Genre!!")
    exit(0)

# match string value of genre with its respective id
genrenum = genre_df.loc[genre]["encoded"]

# match track name
target_vec_df = df_copy.loc[df_copy["track_name"] == trackname ,['popularity', 'explicit', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre']].reset_index()
target_vec_df.drop("index", axis=1, inplace=True)

# match track genre
target_vec_df = target_vec_df.loc[target_vec_df["track_genre"] == genrenum,['popularity', 'explicit', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre']]

# drop the track name - as we dont need it anymore
df_copy.drop("track_name", axis=1, inplace=True)

# Call the annoy function
nearest_neighbours = annoynn(df_copy,target_vec_df)

# Values for nearest neighbours and returning the songs
nearest_labels = df.iloc[nearest_neighbours].sort_values(by='popularity', ascending=False)
nearest_label_values = df_copy.iloc[nearest_neighbours]

# We want only relevant information about the song
nearest_labels = nearest_labels[["track_name","artists", "album_name"]]
nearest_labels_filter = nearest_labels["track_name"] == trackname
nearest_labels = nearest_labels[~nearest_labels_filter]
# top 5 songs
print(f'Song Recommendations for you:\n')
print(nearest_labels.iloc[:6,:])

# and done!

# Importing necessary modules

from annoy import AnnoyIndex # Model
import pandas as pd # Data manipulation
import numpy as np # Data manipulation
from sklearn.preprocessing import LabelEncoder # Encoding categorical vals
import streamlit as st # User Interface
import warnings 
import time
warnings.filterwarnings(action='ignore',category=SyntaxWarning)

#Encoding genres as they are categorical - important part of the model since we will use song name to match values
def encodegenre(dataframe):
    genre_df = pd.DataFrame(data= df_copy["track_genre"].unique(),index=df_copy["track_genre"].unique())
    label = LabelEncoder()
    df_copy["track_genre"] = label.fit_transform(df_copy["track_genre"])
    genre_df["encoded"] = df_copy["track_genre"].unique()
    genre_df.drop(0, axis=1, inplace=True)
    return genre_df

# Model function
def annoynn(dataframe, target_vector_dataframe):
    target_vec_arr = target_vector_dataframe.values.flatten().tolist()
    if(len(target_vec_arr) > 11):
        target_vec_arr = target_vec_arr[:11]

    if(len(target_vec_arr)==0):
        st.write("Error in Song Name/Genre (wrong or empty value).")
        exit(0)

    index = AnnoyIndex(metric="angular", f = dataframe.shape[1])
    for i in range(len(dataframe)):
        vec = dataframe.iloc[i].values
        index.add_item(i, vec)
    index.build(n_trees=200)

    k=500

    nearest_neighbours = index.get_nns_by_vector(target_vec_arr, k)

    nearest_neighbours = [idx for idx in nearest_neighbours if dataframe.iloc[idx]['track_genre'] == genrenum]
    return nearest_neighbours

# Streamlit
st.title("Song Recommendation System")
st.markdown("A Simple Bot that recommends songs based on your song.....")

# Reading the dataframe
df = pd.read_csv("SongsCleaned.csv")
df.drop("dummy", axis=1, inplace=True)

# Creating copy for manipulation
df_copy = df.drop(["Unnamed: 0","track_id","mode","artists", "album_name","duration_ms","time_signature","key","loudness"], axis=1)
df_copy["explicit"] = df_copy["explicit"].astype(int)

# Encoding genres
genre_df = encodegenre(df_copy)

# Get user input for name of song and genre
trackname = st.text_input("Enter Song: ") 
genre = st.selectbox("Pick your genre: ", ["folk", "Other","rock","miscellaneous","metal","jazz","pop","dance","rap","rnb","world"])

# Spinner
with st.spinner('Please wait while the bot searches for the best songs!!'):
    genrenum = genre_df.loc[genre]["encoded"]

    target_vec_df = df_copy.loc[df_copy["track_name"] == trackname ,['popularity', 'explicit', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre']].reset_index()
    target_vec_df.drop("index", axis=1, inplace=True)

    target_vec_df = target_vec_df.loc[target_vec_df["track_genre"] == genrenum,['popularity', 'explicit', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre']]


    df_copy.drop("track_name", axis=1, inplace=True)

    nearest_neighbours = annoynn(df_copy,target_vec_df)


    nearest_labels = df.iloc[nearest_neighbours].sort_values(by='popularity', ascending=False)
    nearest_label_values = df_copy.iloc[nearest_neighbours]

    nearest_labels = nearest_labels[["track_name","artists", "album_name"]]
    nearest_labels_filter = nearest_labels["track_name"] == trackname
    nearest_labels = nearest_labels[~nearest_labels_filter]
    nearest_labels.reset_index(inplace=True, drop=True)
    st.write(f'Song Recommendations for you:\n')
    st.write(nearest_labels.iloc[:5,:])

    st.write(f'More Songs from the artist: ')
    condition = df.loc[df["track_name"] == trackname, ["artists", "track_genre"]]
    condition = condition.loc[condition["track_genre"] == genre, "artists"].iloc[0]
    st.write(f'{condition}')

    songs_from_artist = df.loc[df['artists'] == condition, ["track_name","artists", "album_name","popularity"]]
    songs_from_artist = songs_from_artist.sort_values(by='popularity', ascending=False)
    songs_from_artist.drop("popularity",axis=1, inplace=True)

    songs_from_artist = songs_from_artist[songs_from_artist["track_name"] != trackname]
    songs_from_artist.reset_index(inplace=True, drop=True)

    if(songs_from_artist.shape[0] ==0):
        st.error("Unfortunately, there are no other songs from the artist in the dataset.")
    else:
        st.write(songs_from_artist.iloc[:5,:])

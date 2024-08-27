from annoy import AnnoyIndex
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import warnings
warnings.filterwarnings(action='ignore',category=SyntaxWarning)

# There are 125 different genres. Encode them all using labelencoder, so that the model can use it to recommend songs
# from similar genre
def encodegenre(dataframe):
    genre_df = pd.DataFrame(data= df_copy["track_genre"].unique(),index=df_copy["track_genre"].unique())
    label = LabelEncoder()
    df_copy["track_genre"] = label.fit_transform(df_copy["track_genre"])
    genre_df["encoded"] = df_copy["track_genre"].unique()
    genre_df.drop(0, axis=1, inplace=True)
    return genre_df

# Function to build annoy index.
def annoynn(dataframe, target_vector_dataframe):

    # Convert the target vector from a dataframe to array.
    target_vec_arr = target_vector_dataframe.values.flatten().tolist()
    
    # On the very rare off chance that there are two songs with same data, take only the first song, by stopping at
    # 11 features.
    if(len(target_vec_arr) > 10):
        target_vec_arr = target_vec_arr[:10]
    
    # If the array is empty, that means that there is some error in the data entered.
    if(len(target_vec_arr)==0):
        st.write("Error in Song Name/Genre/Artist (wrong or empty value).")
        exit(0)

    # Build the annoy index for the cleaned dataframe
    index = AnnoyIndex(metric="angular", f = dataframe.shape[1])
    
    # Add vectors and respective index to model
    for i in range(len(dataframe)):
        vec = dataframe.iloc[i].values
        index.add_item(i, vec)
    
    # Build the index
    index.build(n_trees=200)

    # We are finding 500 nearest neighbours
    k=700

    # Get the vectors of the nearest neighbours
    nearest_neighbours = index.get_nns_by_vector(target_vec_arr, k)

    # Locate nearest neighbours with same genre and return them
    nearest_neighbours = [idx for idx in nearest_neighbours if dataframe.iloc[idx]['track_genre'] == genrenum]
    return nearest_neighbours

if __name__ == '__main__':
    # Streamlit !
    st.title("Song Recommendation System")
    st.markdown("Machine learning with vectors to find similar songs!")

    # Read the dataset
    df = pd.read_csv("Original_Songs_Cleaned.csv")

    df_copy = df.drop(["unnamed","Unnamed: 0","track_id","time_signature","mode", "album_name","duration_ms","key","loudness"], axis=1)

    # hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv - original dataset

    # Encode all the different genres
    genre_df = encodegenre(df_copy)

    # Get user input for song name, artist (Must match the value in dataset)
    trackname = st.text_input("Enter Song: ") 
    artist = st.text_input("Enter Artist: ")

    # Create target dataframe containing numerical values accirding to song and artist name
    target_vec_df = df_copy.loc[df_copy['artists'] == artist, ['popularity','artists', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre','track_name']]
    target_vec_df = target_vec_df.loc[target_vec_df["track_name"] == trackname ,['popularity','artists', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre']].reset_index()

    # If user presses enter without filling both fields
    try:
        genrenum = target_vec_df.loc[:,'track_genre'].iloc[0]   
    except IndexError:
        st.write("Enter Song Name and artist")

    # Spinner 
    with st.spinner('Please wait while the bot searches for the best songs!!'):

        # Drop another unwanted field
        target_vec_df.drop("index", axis=1, inplace=True)

        # I have no clue why i added this again but im afraid to remove 
        target_vec_df = target_vec_df.loc[target_vec_df["artists"] == artist,['popularity', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'track_genre']]

        # remove non numerical vals
        df_copy.drop(["track_name", "artists", "explicit"], axis=1, inplace=True)

        # Return nearest neighbours
        nearest_neighbours = annoynn(df_copy,target_vec_df)

        # Return numerical vals of nearest neighbours
        nearest_labels = df.iloc[nearest_neighbours].sort_values(by='popularity', ascending=False)
        nearest_label_values = df_copy.iloc[nearest_neighbours].sort_values(by='popularity', ascending=False)

        # Only display track name, artists and album name 
        nearest_labels = nearest_labels[["track_name","artists", "album_name"]]

        # We dont need track name in input to be displayed again as recommended
        nearest_labels_filter = nearest_labels["track_name"] == trackname
        nearest_labels = nearest_labels[~nearest_labels_filter]

        # Reset index 
        nearest_labels.reset_index(inplace=True, drop=True)

        # Display songs
        st.write(f'Songs you may like:\n')
        st.write(nearest_labels.iloc[:5,:])
        print(nearest_label_values.iloc[:5,:])

        # More songs from the artist from the dataset  
        st.write(f'More Songs from the artist: ')

        # This line is important because it finds exact artist
        # Eg: There are multiple artists who have made a song called "Without me"
        # This Basic code filters out the exact artist
        # And prints it/ uses it to print more songs from them
        condition = df.loc[df["track_name"] == trackname, ["artists", "track_genre"]]
        condition = condition.loc[condition["artists"] == artist, "artists"].iloc[0]
        st.write(f'{condition}')

        # Prints more songs from the artist
        songs_from_artist = df.loc[df['artists'] == condition, ["track_name","artists", "album_name","popularity"]]
        songs_from_artist = songs_from_artist.sort_values(by='popularity', ascending=False)
        songs_from_artist.drop("popularity",axis=1, inplace=True)

        # Without the input track
        songs_from_artist = songs_from_artist[songs_from_artist["track_name"] != trackname]
        songs_from_artist.reset_index(inplace=True, drop=True)

        # Dataset only has 80,000 songs, so there may not be many songs from the same artist 
        if(songs_from_artist.shape[0] ==0):
            st.error("Unfortunately, there are no other songs from the artist in the dataset.")
        else:
            st.write(songs_from_artist.iloc[:5,:])
    # Thats all !! 

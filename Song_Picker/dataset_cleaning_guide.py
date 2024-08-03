import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings(action='ignore',category=SyntaxWarning)

def encodegenre(dataframe):
    genre_df = pd.DataFrame(data= df_copy["track_genre"].unique(),index=df_copy["track_genre"].unique())
    label = LabelEncoder()
    df_copy["track_genre"] = label.fit_transform(df_copy["track_genre"])
    genre_df["encoded"] = df_copy["track_genre"].unique()
    genre_df.drop(0, axis=1, inplace=True)
    return genre_df


df = pd.read_csv("SongsCleaned.csv")
df.drop("dummy", axis=1, inplace=True)

# artist names can be in lowercase, remove featuring artists
df["artists"] = df["artists"].str.replace(';',', ').str.split(r'\s*\,\s*',n=1).str[0]

# remove feature names from song because you cant memorize it all
# also convert the song names to lowercase because im too lazy to search for songs with case sensitivity
df["track_name"] = df["track_name"].str.split(r'\s*\(\s*',n=1).str[0]
df["track_name"] = df["track_name"].str.lower()

duplicates = df.duplicated(subset= ["track_name", "artists"], keep= 'first')
df = df[~duplicates]

# drop unwanted stuff
df_copy = df.drop(["Unnamed: 0","track_id","mode","artists", "album_name","duration_ms","time_signature","key","loudness"], axis=1)
df_copy["explicit"] = df_copy["explicit"].astype(int)

genre_df = encodegenre(df_copy)

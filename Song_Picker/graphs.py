import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("Original_Songs_Cleaned.csv")

df_copy = df.drop(["unnamed","Unnamed: 0","track_id","artists","album_name","track_name","duration_ms","track_genre"], axis=1)
x = lambda x: 1 if x == True else 0
df["explicit"] = df["explicit"].apply(x)

print(df_copy.columns)

'''
sns.histplot(x = "time_signature", data= df_copy)
plt.show()
# Time_Signature is biased
'''

elements_to_plot = ['popularity', 'explicit', 'danceability', 'energy', 'key', 'loudness',
       'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo', 'time_signature']

# Explicit, key, mode, time_signature are not normally distributed, can be removed
fig, axes = plt.subplots(7,2)

for i in range(14):
    sns.histplot(ax=axes.flatten()[i], x = df_copy[elements_to_plot[i]])
plt.show()

sns.heatmap(data=df_copy.corr(), cmap= 'coolwarm', annot=True)
plt.show()
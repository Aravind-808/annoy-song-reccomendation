>  # Very Basic demonstration of how the ANNOY (Approximate Nearest Neighbours) algorithm works, by an absolute beginner.
> - ANNOY is the stuff they use in spotify to recommend you songs based on the one you listen to. It converts the target datapoints into vectors, and calculates the closest vectors to the target datapoint, in the dataset (or thats what i understood).
>
> - So I was thinking of a way to learn how it worked, and what better way than to make your own song recommending system (again, very basic)
>   
> - I used an online dataset, cleaned it (very basic cleaning) and then used that as my own dataset for the operation - as the old one was a bit shabby.
>   If you want the online dataset to perform the cleaning yourself, here you go!:
>   
>   [hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv](url)
>
> - A small disclaimer: Before you use this dataset, make sure to install the following packages.
>   ```
>   pip install fsspec
>   pip install huggingface_hub
>   ```
> - And before you start with the code execution, and I think this goes without saying: Install the following packages!
>   ```
>   from annoy import AnnoyIndex
>   import pandas as pd
>   import numpy as np
>   from sklearn.preprocessing import LabelEncoder
>   ```
> - A final disclaimer before you begin: A LOT of the songs that are in the dataset are classified wrongly. In the original dataset, there were 125 genres. For the sake of simplification, I grouped them into 10 genres.
>   and that came at the cost of putting some songs in the wrong genre. This is because my approach was to approximate songs based on their genre and a wide range of metrics you'll get to know are used in the dataset.
>   That is also the reason i have linked the original dataset - so that you can come up with your own way of doing it.
>
> - Thats it! Bye bye !!


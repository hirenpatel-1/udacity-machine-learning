import requests

print (dir(requests))
# The direct link to the Kaggle data set
data_url = 'https://www.kaggle.com/c/zillow-prize-1/download/properties_2016.csv.zip'

# The local path where the data set is saved.
local_filename = "properties_2016.csv.zip"

# Kaggle Username and Password
kaggle_info = {'UserName': "mca.hiren@gmail.com", 'Password': "splender0543"}

# Attempts to download the CSV file. Gets rejected because we are not logged in.
r = requests.get(data_url)

# Login to Kaggle and retrieve the data.
r = requests.post(r.url, data = kaggle_info, stream=True)

# Writes the data to a local file one chunk at a time.
f = open(local_filename, 'w')
for chunk in r.iter_content(chunk_size = 512 * 1024): # Reads 512KB at a time into memory
    if chunk: # filter out keep-alive new chunks
        f.write(chunk)
f.close()
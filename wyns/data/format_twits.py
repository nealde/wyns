import pandas as pd
data = pd.read_csv('tweets.txt', delimiter='~~n~~', engine='python',
                   names=['text', 'long', 'lat'])
print(data.head())
# with open('tweets.txt') as f:
#    for line in f.readline

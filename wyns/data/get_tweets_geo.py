#import jsonpickle
import sys
import tweepy
import pandas as pd

#  Use to get tweets in a way that bypasses twitters weird rules
#  Should be able to run on a build node on hyak - havent tested

#  Consumer key, consumer secret, access token, access secret.
#  Unique to each person. Read Wes' API notebook for more
API_KEY = 'IPbYoAbOUR1URWvXWeNwQNnZD'
API_SECRET = 'goN7XnztVpn6CgkEAAxU9GOVSwbUYwjuFC0ChXdxjWBhRrYZcj'
access_token = '506759494-rt09qdTZGlGH8WkBDd5M8Vgr6eGbZtlxQVaEH7hA'
access_token_secret = 'k6tPQuDCnqIf25Ethn6mtZ4pTAoncEufAIy8EVujP2JF2'

auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True)

if (not api):
    print("Can't Authenticate")
    sys.exit(-1)

searchQuery = 'climate change'
maxTweets = 1800  # Some arbitrary large number)
tweetsPerQry = 100  # Max the API permits per query
fName = 'tweets_geo.txt'  # Stores tweets in text as well as a json file

#  Below basically prevents pulling duplicate tweets (I think)
sinceId = None
max_id = -1

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))
#ff = []
with open(fName, 'a') as f:
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry, tweet_mode='extended')
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            since_id=sinceId, tweet_mode='extended')
            else:
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1), tweet_mode='extended')
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1),
                                            since_id=sinceId, tweet_mode='extended')
            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                if tweet.geo is not None:
#                    print(tweet.geo.coordinates[0])

                    try:
                        f.write(str(tweet.full_text)+'~~n~~'+str(tweet.geo.coordinates[0])+'~~n~~'+str(tweet.geo.coordinates[1])+'~~n~~'+str(tweet.place.full_name)+'\n')
                        tweetCount += 1
                    except:
                        continue
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
#data = pd.DataFrame(ff,columns=['text','long','lat'])
#data.to_csv("tweets.csv",sep='~!')
#            break

print("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))

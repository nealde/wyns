import sys
import tweepy
from tweet import API_KEY, API_SECRET

auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True)

if not api:
    print("Can't Authenticate")
    sys.exit(-1)

searchQuery = 'climate change'
maxTweets = 1800  # Some arbitrary large number)
tweetsPerQry = 100  # Max the API permits per query
fName = 'tweets_geo.txt'  # Stores tweets in a json file

#  Below prevents pulling duplicate tweets
sinceId = None
max_id = -1

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))
with open(fName, 'a') as f:
    while tweetCount < maxTweets:
        try:
            if max_id <= 0:
                if not sinceId:
                    new_tweets = api.search(q=searchQuery,
                                            count=tweetsPerQry,
                                            tweet_mode='extended')
                else:
                    new_tweets = api.search(q=searchQuery,
                                            count=tweetsPerQry,
                                            since_id=sinceId,
                                            tweet_mode='extended')
            else:
                if not sinceId:
                    new_tweets = api.search(q=searchQuery,
                                            count=tweetsPerQry,
                                            max_id=str(max_id - 1),
                                            tweet_mode='extended')
                else:
                    new_tweets = api.search(q=searchQuery,
                                            count=tweetsPerQry,
                                            max_id=str(max_id - 1),
                                            since_id=sinceId,
                                            tweet_mode='extended')
            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                if tweet.geo is not None:
                    try:
                        f.write(str(tweet.full_text + '~~n~~' +
                                    str(tweet.geo.coordinates[0]) + '~~n~~' +
                                    str(tweet.geo.coordinates[1]) + '~~n~~' +
                                    str(tweet.place.full_name) + '\n'))
                        tweetCount += 1
                    except:
                        continue
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
#           break

print("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))

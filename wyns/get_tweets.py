import sys
import tweepy

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
lang = 'en'
geo = '43.913723261972855,-72.54272478125,150km'
maxTweets = 1  # Some arbitrary large number)
tweetsPerQry = 100  # Max the API permits per query
fName = 'data/tweets.txt'  # Stores tweets in json file

#  Below basically prevents pulling duplicate tweets (I think)
sinceId = None
max_id = -1

tweetCount = 0
print("Downloading max {0} tweets".format(maxTweets))

with open(fName, 'w') as f:
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, 
                                            lang=lang,
                                            geocode=geo,
                                            count=tweetsPerQry,
                                            tweet_mode='extended')
                else:
                    new_tweets = api.search(q=searchQuery,
                                            lang=lang,
                                            geocode=geo,
                                            count=tweetsPerQry,
                                            tweet_mode='extended',
                                            since_id=sinceId)
            else:
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, 
                                            lang=lang,
                                            geocode=geo,
                                            count=tweetsPerQry,
                                            tweet_mode='extended',
                                            max_id=str(max_id - 1))
                else:
                    new_tweets = api.search(q=searchQuery,
                                            lang=lang,
                                            geocode=geo,
                                            count=tweetsPerQry,
                                            tweet_mode='extended',
                                            max_id=str(max_id - 1),
                                            since_id=sinceId)
            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                print(tweet.entities)
                # f.write(json.encode(tweet._json, unpicklable=False) +
                #        '\n')
            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            break

print("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))

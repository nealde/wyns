import twitter
import pandas as pd
import json
import configparser
import os

class TweetRc(object):
    def __init__(self):
        self._config = None

    def GetConsumerKey(self):
        return self._GetOption('consumer_key')

    def GetConsumerSecret(self):
        return self._GetOption('consumer_secret')

    def GetAccessKey(self):
        return self._GetOption('access_key')

    def GetAccessSecret(self):
        return self._GetOption('access_secret')

    def _GetOption(self, option):
        try:
            return self._GetConfig().get('Tweet', option)
        except:
            return None

    def _GetConfig(self):
        if not self._config:
            self._config = configparser.ConfigParser()
            self._config.read(os.path.expanduser('./.tweetrc')) 
        return self._config

tw = TweetRc()
api=twitter.Api(consumer_key=tw.GetConsumerKey(),
consumer_secret=tw.GetConsumerSecret(),
access_token_key=tw.GetAccessKey(),
access_token_secret=tw.GetAccessSecret())

with open("tweet_feed.json", 'a') as f:
    for line in api.GetStreamFilter(track=['global warming', 'climate change',
                                           'sustainability', 'pollution'],
                                   languages=['en'], filter_level=['low']):
        f.write(json.dumps(line))
        f.write("\n")

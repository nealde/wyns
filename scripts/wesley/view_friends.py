#!/usr/bin/env python

# Copyright 2016 The Python-Twitter Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------------------
# Change History
# 2010-10-01
#   Initial commit by @jsteiner207
#
# 2014-12-29
#   PEP8 update by @radzhome
#
# 2016-05-07
#   Update for Python3 by @jeremylow
#

from __future__ import print_function
import twitter

api=twitter.Api(consumer_key="lJldK3X9BmSTknab2E1M4TnQu",
consumer_secret="kJfXHUXpSySRZmLsozOR4hrj0BwemMekPCXtyMhUSROFTaU7jv",
access_token_key="3018876422-38DSGHL2rG2XYvER2yF3G2mBSEjaurFr8EgVQ1c",
access_token_secret="OWR5txTPUBhIpKfnEhgSpVgN0eW2MBvoj7vsGFaqfTcZE")


# Create an Api instance.

users = api.GetFriends()

print([u.screen_name for u in users])

"""
Pulls most recent tweets from Donald Trump's twitter account
and uses a Classifier to predict if they are Trump or Staff

Alexander Li (afl59), Rohan Patel (rp442)
1/10/19
"""
import twitter

def getTweets():
    api = twitter.Api(consumer_key='mxSTcGP1nh6jmDG7MCB0BuOex',
      consumer_secret='6KO0vUmpeG04xzyNz5DSOXPvRKjPfAMcblGMhP6pyDrwQdPmlM',
      access_token_key='491599498-QrCGgzC8scH15NbDq3qIbmNCnu9meo8Rc0u2L7qA',
      access_token_secret='iR5tDCtehji5iWnEL7wrLsFItEepwS8bUHOmpKGjlcoUm',
      tweet_mode="extended")

    t = api.GetUserTimeline(screen_name="realDonaldTrump", include_rts=False, count=40)
    tweets = [i.AsDict() for i in t]
    test = []
    for t in tweets:
        #print(t['id'], t['text'])
        test.append(t['full_text'])

    return test

def predictTweets(tweets, model, vec, selector):
    tweets = vec.transform(tweets).toarray()
    tweets = selector.transform(tweets).astype('float32')
    preds = model.predict(tweets).tolist()
    predsNumber = preds[:]
    predsUrl = preds[:]
    for i in range(len(preds)):
        if preds[i] > 0:
            preds[i] = "Donald J. Trump"
            predsUrl[i] = "https://pbs.twimg.com/profile_images/874276197357596672/kUuht00m_400x400.jpg"
        else:
            preds[i] = "White House Staff"
            predsUrl[i] = "https://abs.twimg.com/sticky/default_profile_images/default_profile_400x400.png"
    return preds, predsNumber, predsUrl

#print(test)
# print('training')
# models, vec, selector = training()
# print('testing')
# test = vec.transform(test).toarray()
# test = selector.transform(test).astype('float32')
# for m in range(len(models)):
#      pred = models[m].predict(test)
#      print('model: ' + str(m))
#      print(pred)

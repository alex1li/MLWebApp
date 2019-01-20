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
            predsUrl[i] = "data:image/png;base64,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"
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

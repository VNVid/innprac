#!/usr/bin/env python3
from code_to_test import *

########################################################
# sentiment prediction analysis
# positive phrases
pos = ["Ann is cool",
       "Den is cool",
       "He's great!",
       "I like apples",
       "I like bananas"]

# negative phrases
neg = ["I don't like bananas",
       "I have too many things to do",
       "I have too much work to do",
       "it's disgusting",
       "the service is disgusting"]

# neutral
neu = ["She's working",
       "red carpet",
       "green carpet"]


pos_pred = sentiment_analyzer.predict(pos)
neg_pred = sentiment_analyzer.predict(neg)
neu_pred = sentiment_analyzer.predict(neu)


def compare_pred(preds: list, ans: str):
    wrong_cnt = 0
    for pred in preds:
        if pred.output != ans:
            wrong_cnt += 1
    return wrong_cnt


########################################################
# emotion prediction analysis
phrases_em = ["Wow, I like the present",                       # joy
              "The dress looks ridiculous",                    # disgust
              "I didn't expect that in any way.",              # surprise
              "I miss him",                                    # sadness
              "She misses him",                                # sadness
              "all friends missed her",                        # sadness
              "I'm worried about tomorrow's exam",             # fear
              "Everyone is worried about tomorrow's exam"]     # fear

correct_labels = ['joy', 'disgust', 'surprise',
                  'sadness', 'sadness', 'sadness', 'fear', 'fear']

emotion_prediction = emotion_analyzer.predict(phrases_em)

wrong_cnt = 0
for i in range(len(emotion_prediction)):
    if emotion_prediction[i].output != correct_labels[i]:
        wrong_cnt += 1


#########################################################
# Printing results
print("\n\nSentiment prediction analysis:")
print("number of wrong predictions {}".format(compare_pred(pos_pred, 'POS') +
                                              compare_pred(neg_pred, 'NEG') +
                                              compare_pred(neu_pred, 'NEU')))
print("Emotion prediction analysis:")
print(f"number of wrong predictions {wrong_cnt}")

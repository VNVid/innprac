#!/usr/bin/env python3
from code_to_test import *

# for sentiment
text_file = open("tweeteval/datasets/sentiment/test_text.txt", "r")
label_file = open("tweeteval/datasets/sentiment/test_labels.txt", "r")

labels = []
predictions = []

label_dict = {'0': "NEG", '1': "NEU", '2': "POS"}

while True:
    line = text_file.readline()
    label = label_file.readline()
    if not line or not label:
        break

    predictions.append(sentiment_analyzer.predict(line).output)
    labels.append(label_dict[label[0]])

text_file.close
label_file.close

cm = confusion_matrix(labels, predictions, labels=['NEG', 'NEU', 'POS'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['NEG', 'NEU', 'POS'])
disp.figure_.savefig('sentiment.png', dpi=300)

# for emotion
text_file = "SILICONE-benchmark/dyda/test.csv"
df = pd.read_csv(text_file, delimiter=',')

labels = []
predictions = []


label_dict = {"anger": "anger",
              "disgust": "disgust",
              "fear": "fear",
              "happiness": "joy",
              "no emotion": "others",
              "sadness": "sadness",
              "surprise": "surprise"}

for i, row in df.iterrows():
    predictions.append(emotion_analyzer.predict(row['Utterance']).output)
    labels.append(label_dict[row['Emotion']])


labels_name = ["anger", "disgust", "fear",
               "joy", "sadness", "surprise", "others"]
cm = confusion_matrix(labels, predictions, labels=labels_name)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels_name)
disp.figure_.savefig('emotion.png', dpi=300)

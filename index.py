from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer

# train model
positive_texts = [
    "we love you",
    "they love us",
    "you are good",
    "he is good",
    "they love mary"
]

negative_texts =  [
    "we hate you", 
    "they hate us",
    "you are bad",
    "he is bad",
    "we hate mary"
]

# test model
test_texts = [
    "they love mary",
    "they are good",
    "why do you hate mary",
    "they are almost always good",
    "we are very bad"
]

training_texts = negative_texts + positive_texts
training_labels = ["negative"] * len(negative_texts) + ["positive"] * len(positive_texts)
# converting text into number
vectorizer = CountVectorizer()
vectorizer.fit(training_texts)
print(vectorizer.vocabulary_)
# vector transform
training_vectors = vectorizer.transform(training_texts)
testing_vectors = vectorizer.transform(test_texts)
# Classifier
classifier = tree.DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)
predictions = classifier.predict(testing_vectors)
print(predictions)


tree.export_graphviz(
    classifier,
    out_file='tree.dot',
    feature_names=vectorizer.get_feature_names(),
) 

# same work but different logic
def manual_classify(text):
    if "hate" in text:
        return "negative"
    if "bad" in text:
        return "negative"
    return "positive"

predictions = []
for text in test_texts:
    prediction = manual_classify(text)
    predictions.append(prediction)
print("I am Here")
print(predictions)

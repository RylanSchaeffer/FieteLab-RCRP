import numpy as np
import sklearn.datasets
import sklearn.feature_extraction.text

# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
exp_dir = 'exp_05_language_modeling'

categories = None
categories = ['soc.religion.christian', 'comp.graphics', 'sci.med']


twenty_train = sklearn.datasets.fetch_20newsgroups(
    subset='train',
    categories=categories,
    shuffle=True,
    random_state=0)
twenty_test = sklearn.datasets.fetch_20newsgroups(
    subset='test',
    categories=categories,
    shuffle=True,
    random_state=0)

train_class_names = np.array(twenty_train.target_names)
train_targets = twenty_train.target
train_targets_class_names = train_class_names[train_targets]

# count the word occurrences
count_vect = sklearn.feature_extraction.text.CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_test_counts = count_vect.transform(twenty_test.data)

# convert counts to term frequencies (tf)
# a better version of tf is tf-idf (Term Frequency times Inverse Document Frequency)
# tf-idf downscales words that occur commonly e.g. "the", "a", "said"
tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
print(X_train_tfidf.shape)

temp = X_train_tfidf.toarray()


from sklearn.naive_bayes import MultinomialNB

text_clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
predicted = text_clf.predict(X_test_tfidf)
np.mean(predicted == twenty_test.target)

print(10)

# https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py

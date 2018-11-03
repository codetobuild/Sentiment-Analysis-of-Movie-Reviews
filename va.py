import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# to remove Deprication warning
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# Importing the dataset
dataset = pd.read_csv(r"C:\Users\Saurabh\Videos\dataset\movie_review.csv")
#print(dataset.head())

#Cleaning the dataset
dataset.drop(['fold_id','cv_tag','html_id','sent_id'],axis=1,inplace=True)
#print(dataset.head())


#Checking for the null values
#print(dataset.isna().sum())


#Tokenization using Regular expression
tokenizer = RegexpTokenizer("[a-zA-Z]+")
#print(dataset['text'][0])

sw = set(stopwords.words('English'))
ps = PorterStemmer()
corpus = []

for i in range(0,len(dataset['text'])):
	review = dataset['text'][i].lower()
	#print(review)
	review = tokenizer.tokenize(dataset['text'][i])
	#print(review)

	#Removing the stopwords and stemming using Porter Stemmer
	review = [ps.stem(word) for word in review if word not in sw]
	review = ' '.join(review)
	corpus.append(review)


#Vectorization
cv = CountVectorizer(max_features=15000)
X = cv.fit_transform(corpus).toarray()

Y = dataset.iloc[:,1].values

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)

#Fitting the Naive  Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train,Y_train)


#Predicting the Test set results
Y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
print(cm)

# Accuracy
acc = accuracy_score(Y_test,Y_pred)
print(acc)























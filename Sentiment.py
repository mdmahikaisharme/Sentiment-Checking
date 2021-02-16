# Import
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import json, os

# Reviews List & Class
LReviews = []
class CReviews():
    def __init__(self, text, rate):
        self.text = text
        self.sentiment = "Negative" if rate <= 2 else "Neutral" if rate == 3 else "Positive"

# Open Date
base = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(base, "Sentiment.json")) as f:
        for i in f:
            _i = json.loads(i)
            LReviews.append(CReviews(_i["reviewText"], _i["overall"]))
except:
    print("-----------------------------------------------------------\n    We Didn't Found Our File Data Of Sentiment Program!    \n-----------------------------------------------------------")

# ReviewsBox Class
class CReviewsBox():
    def __init__(self, reviews):
        self.reviews = reviews
        self.box()
    def text(self):
        return [i.text for i in self.reviews]
    def sentiment(self):
        return [i.sentiment for i in self.reviews]
    def box(self):
        negative = list(filter(lambda x: x.sentiment == "Negative", self.reviews))
        positive = list(filter(lambda x: x.sentiment == "Positive", self.reviews))
        positive = positive[:len(negative)]
        self.reviews = negative +positive

# Train Test Split
""" May Be We Don't Need Test! """
train, test = train_test_split(LReviews, test_size=0.2, random_state=42)
train = CReviewsBox(train)
test = CReviewsBox(test)
train_x = train.text()
train_y = train.sentiment()
test_x = test.text()
test_y = test.sentiment()

# Vectorizer
vect = TfidfVectorizer()
train_x_vect = vect.fit_transform(train_x)

# Classifier
clf_svm = SVC(kernel="linear")
clf_svm.fit(train_x_vect, train_y)

# Check Your Sentiment
print(clf_svm.predict(vect.transform([str(input("Comment: "))])))
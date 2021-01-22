import nltk
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

#download packages
# nltk.download('punkt')
# nltk.download('wordnet')

answers = open('dog.txt', 'r', errors='ignore')
answers = answers.read()
answers = answers.lower().encode('ASCII','ignore').decode('ASCII')

sent_token = nltk.sent_tokenize(answers)
word_token = nltk.word_tokenize(answers)
lemmer = nltk.stem.WordNetLemmatizer()

def lemmer_tokens(tokens):
    return[lemmer.lemmatize(token) for token in tokens]

remove_punct = dict((ord(punct), None) for punct in string.punctuation)

def lemmer_normalize(text):
    return lemmer_tokens(nltk.word_tokenize(text.lower().translate(remove_punct)))

def response(user_response):
    answer_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=lemmer_normalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_token)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        answer_response = answer_response + "<Invalid Question>"
        return answer_response
    else:
        answer_response = answer_response + sent_token[idx]
        return answer_response

user_response = input("Enter question here >>>")
user_response = user_response.lower()
sent_token.append(user_response)
word_token = word_token + nltk.word_tokenize(user_response)
final_words = list(set(word_token))
print(response(user_response))
print()

sent_token.remove(user_response)
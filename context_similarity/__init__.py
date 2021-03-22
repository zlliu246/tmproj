import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def select_context(question, vectorizer, context_vectors, contexts):
    question_vector = vectorizer.transform([question])
    cossim = cosine_similarity(context_vectors, question_vector)

    indices = cossim[:,0].argsort()[::-1][:3]
    return "\n".join([contexts[i] for i in indices])


if __name__ == "__main__":
    args = pickle.load(open("context_vectors.sav", "rb"))
    
    qn = "who is beyonce and what is she up to?"
    con = select_context(qn, *args)

    print(con)
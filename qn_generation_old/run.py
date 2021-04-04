import pickle
from time import time
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def get_closest_qa_pair(target_qn, vectorizer, question_vectors, questions, answers):
    target_vector = vectorizer.transform([target_qn])
    cossim = cosine_similarity(question_vectors, target_vector)

    i = np.argmax(cossim)

    qn, an = questions[i], answers[i]

    if type(an) == list:
        an = [i for i in an if i["correct"]][0]["answer"]

    return qn, an


if __name__ == "__main__":

    vectorizer, question_vectors, existing_questions, existing_answers = pickle.load(open("model.sav", "rb"))

    while True:

        question = input("input question here >>>")

        closest_qn, answer = get_closest_qa_pair(question, vectorizer, question_vectors, existing_questions, existing_answers)

        print(f"input question: {question}\nclosest question: {closest_qn}\nanswer: {answer}\n")


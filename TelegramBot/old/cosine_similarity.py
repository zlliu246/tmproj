from qn_generation.run import get_closest_qa_pair
import pickle

class CosineSimilarityHandler():

    def __init__(self):
        self.args = pickle.load(open("qn_generation/model.sav", "rb"))


    def handle(self, question):

        q,a = get_closest_qa_pair(question, *self.args)
        return f"cossim: {a}"
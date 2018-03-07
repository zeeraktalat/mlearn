class Vectorisation(object):
    """ This is an issue """
    def __init__(self, X, y, vectoriser):
        """ Initialise vectoriser """
        self.vect = vectoriser

    @property
    def vectoriser(self):
        return self.vect

    @vectoriser.setter
    def vectoriser(self, vect):
        self.vect = vectoriser

    def train(self, X):
        return self.vect.fit_transform(X)

    def test(self, X):
        return self.vect.transform(X)

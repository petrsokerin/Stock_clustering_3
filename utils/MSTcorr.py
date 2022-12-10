from sklearn.cluster import AgglomerativeClustering
import numpy as np

class MSTcorrClustering:

    def __init__(self, n_clusters, linkage='complete'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, data):
        corr = np.corrcoef(data)
        D = 1 - corr ** 2
        clst = AgglomerativeClustering(n_clusters=self.n_clusters, affinity='precomputed',
                                       linkage=self.linkage)
        clst.fit(D)
        self.labels_ = clst.labels_
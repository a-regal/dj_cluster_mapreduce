import rtree
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from mrjob.job import MRJob
from mrjob.step import MRStep

tree = rtree.index.Index('./tree/moons')
eps = 0.72
min_pts = 4
r = 0.1

def knn(neighborhood, point, radius):
    ids = []
    for ix, neighbor in enumerate(neighborhood):
        p = np.array([neighbor.bbox[0], neighbor.bbox[1]])
        pairs = np.vstack([point,p])
        dist = haversine_distances(pairs)[0,1]
        if dist < r:
            ids.append(neighbor.id)
        else:
            pass
    return ids

class MRDJCluster(MRJob):
    def mapper(self, key, value):
        idx, lat, lon = value.split(',')
        trace = (float(lon), float(lat))

        #Compute tree knn
        neighborhood = list(tree.nearest(trace, num_results=min_pts, objects=True))

        #Ensure they are within r
        neighborhood_ = knn(neighborhood, trace, r)

        if len(neighborhood_) < min_pts:
            yield None, [-1]
        else:
            yield None, neighborhood_

    def reducer(self, key, values):
        values = list(values)
        #List of sets
        clusters = []
        for neighborhood in values:
            #Control variables to check if there are changes among the clusters
            start_l = len(clusters)
            cluster_sizes = [len(cluster) for cluster in clusters]

            #Since clusters start empty an initial cluster is created with the
            #first neighborhood
            if len(clusters) == 0:
                clusters.append(set(neighborhood))
            else:
                for ix, cluster in enumerate(clusters):
                    #Do set intersection to check if a nn is within a cluster
                    if set(neighborhood).intersection(cluster):
                        clusters[ix] = cluster.union(neighborhood)
                        break
                else:
                    clusters.append(set(neighborhood))

        for ix, cluster in enumerate(clusters):
            yield ix, list(cluster)

    def steps(self):
        return [
            MRStep(mapper=self.mapper,
                   reducer=self.reducer)
        ]


if __name__ == '__main__':
    MRDJCluster.run()

import pyspark
import rtree
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from pyspark import SparkContext

#Constants
min_pts = 4
r = 0.1

#Setup tree and load data
tree = rtree.index.Index('./tree/moons')
df = pd.read_csv('moons.csv', header=None)

#Util function for knn calculation
def knn(point, tree, r, min_pts):
    #Convert to tuple, see rtree documentation
    trace = tuple(point)

    #Compute closest points according to the tree. Distance needs to be checked
    #We return the object to access the bounding box of each point
    neighborhood = list(tree.nearest(trace, num_results=min_pts, objects=True))

    #Create a "clean" list of ids
    ids = []
    for ix, neighbor in enumerate(neighborhood):
        #Get point from bbox (4 coords for left, bottom, right, upper), we only  need 2
        p = np.array([neighbor.bbox[0], neighbor.bbox[1]])

        #Stack the point with the bbox to compute the distance matrix with sklearn
        pairs = np.vstack([point,p])
        dist = haversine_distances(pairs)[0,1]

        #Check which of these are within the radius
        if dist < r:
            ids.append(neighbor.id)
        else:
            pass

    if len(ids) < min_pts:
        return [-1]
    else:
        return ids

#Util function to merge clusters
def cluster(values):
    #Initialize clusters
    clusters = []

    #Begin cluster merging
    for neighborhood in values:
        #For the first iteration the first neighborhood is the cluster
        if len(clusters) == 0:
            clusters.append(set(neighborhood))
        else:
            #Verify set intersection for each cluster
            for ix, cluster in enumerate(clusters):
                #If the intersection isn't an empty set
                if set(neighborhood).intersection(cluster):
                    #Assign the cluster in that index as the union of it and the neighborhood
                    clusters[ix] = cluster.union(set(neighborhood))
                    #Break the loop, no more iterations needed
                    break
            #If nothing happend in the loop (the break clause was not encountered)
            else:
                #Append to the cluster list
                clusters.append(set(neighborhood))

#Create spark context
sc = SparkContext()

#Parallelize df values (only x,y coordinate values)
rdd = sc.parallelize(df.values[:,1:].tolist())


mapper = rdd.map(lambda x: (None, knn(x, tree, r, min_pts)))
reducer = mapper.reduceByKey(lambda x: cluster(x))

with open('./results_pyspark.txt', 'w') as f:
    f.write(reducer.collect())

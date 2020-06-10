import pandas as pd
import rtree
from sklearn.datasets import make_moons

#Create moon shaped points with 0.1 variance gaussian noise
data, labels = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)

#Convert to df
df = pd.DataFrame(data)

#Save without header: id, x, y
df.to_csv('moons.csv', header=None)

#Create rtree (without map-reduce)
idx = rtree.index.Index('./tree/moons')

for index, row in df.iterrows():
    #Insert (left, bottom, right, up) bounding box. Since everything is a point
    #we use the same coords
    idx.insert(index, (row[1], row[0], row[1], row[0]))

idx.close()

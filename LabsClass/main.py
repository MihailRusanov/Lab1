import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

data = make_blobs(
    n_samples=100,
    n_features=2,
    centers=3,
    cluster_std=1.5
)

model = KMeans(n_clusters=4)
model.fit(data[0])

print(model.labels_) # - определят № кластера каждой точки
print(model.cluster_centers_) #- центр каждого кластера

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))
ax1.set_title('Наши предсказания')
ax1.scatter(data[0][:, 0], data[0][:, 1], c=model.labels_)
ax2.set_title('Реальные значения')
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1])

plt.show()

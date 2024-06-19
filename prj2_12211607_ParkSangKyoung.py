import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

ratings = pd.read_csv('ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'])

print (ratings)

unique_movie_ids = sorted(ratings['movie_id'].unique())
print(unique_movie_ids)

user_movie_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

km = KMeans(n_clusters=3, random_state=0)

data = user_movie_matrix
km.fit(data)

group1 = data[km.labels_ == 0].copy()
group2 = data[km.labels_ == 1].copy()
group3 = data[km.labels_ == 2].copy()

AU1 = np.sum(group1, axis=0)
AU2 = np.sum(group2, axis=0)
AU3 = np.sum(group3, axis=0)

Avg1 = np.mean(group1, axis=0)
Avg2 = np.mean(group2, axis=0)
Avg3 = np.mean(group3, axis=0)

SC1 = np.sum(group1 > 0, axis=0)
SC2 = np.sum(group2 > 0, axis=0)
SC3 = np.sum(group3 > 0, axis=0)

AV1 = np.sum(group1 >= 4, axis=0)
AV2 = np.sum(group2 >= 4, axis=0)
AV3 = np.sum(group3 >= 4, axis=0)

BC1 = (group1 > 0).rank(axis=1, method='average').sum(axis=0)
BC2 = (group2 > 0).rank(axis=1, method='average').sum(axis=0)
BC3 = (group3 > 0).rank(axis=1, method='average').sum(axis=0)

group1np = group1.values
n = group1np.shape[1]

new_array1 = np.empty((n,n))

for i in range(n):
    this_row = group1np[:, i][:, np.newaxis]  # (3706, 1) 배열로 변경
    result = np.where(group1np < this_row, 1, np.where(group1np > this_row, -1, 0))
    row_sums = result.sum(axis=0)
    row_sums = np.where(row_sums > 0, 1, np.where(row_sums < 0, -1, 0))
    new_array1[:, i] = row_sums

group2np = group2.values
n = group2np.shape[1]

new_array2 = np.empty((n,n))

for i in range(n):
    this_row = group2np[:, i][:, np.newaxis]  # (3706, 1) 배열로 변경
    result = np.where(group2np < this_row, 1, np.where(group2np > this_row, -1, 0))
    row_sums = result.sum(axis=0)
    row_sums = np.where(row_sums > 0, 1, np.where(row_sums < 0, -1, 0))
    new_array2[:, i] = row_sums

group3np = group3.values
n = group3np.shape[1]

new_array3 = np.empty((n,n))

for i in range(n):
    this_row = group3np[:, i][:, np.newaxis]  # (3706, 1) 배열로 변경
    result = np.where(group3np < this_row, 1, np.where(group3np > this_row, -1, 0))
    row_sums = result.sum(axis=0)
    row_sums = np.where(row_sums > 0, 1, np.where(row_sums < 0, -1, 0))
    new_array3[:, i] = row_sums

CR1 = np.sum(new_array1, axis=0)
CR2 = np.sum(new_array2, axis=0)
CR3 = np.sum(new_array3, axis=0)
CR1 = pd.Series(CR1, index=group1.columns)
CR2 = pd.Series(CR2, index=group2.columns)
CR3 = pd.Series(CR3, index=group3.columns)

group1AU1 = AU1.nlargest(10).index
group2AU2 = AU2.nlargest(10).index
group3AU3 = AU3.nlargest(10).index

group1Avg1 = Avg1.nlargest(10).index
group2Avg2 = Avg2.nlargest(10).index
group3Avg3 = Avg3.nlargest(10).index

group1SC1 = SC1.nlargest(10).index
group2SC2 = SC2.nlargest(10).index
group3SC3 = SC3.nlargest(10).index

group1AV1 = AV1.nlargest(10).index
group2AV2 = AV2.nlargest(10).index
group3AV3 = AV3.nlargest(10).index

group1BC1 = BC1.nlargest(10).index
group2BC2 = BC2.nlargest(10).index
group3BC3 = BC3.nlargest(10).index

group1CR1 = CR1.nlargest(10).index
group2CR2 = CR2.nlargest(10).index
group3CR3 = CR3.nlargest(10).index

Group1 = pd.DataFrame(index=['Top1','Top2', 'Top3', 'Top4', 'Top5', 'Top6', 'Top7', 'Top8', 'Top9', 'Top10'], columns=['AU1', 'Avg1', 'SC1', 'AV1', 'BC1', 'CR1'])
Group1['AU1'] = group1AU1
Group1['Avg1'] = group1Avg1
Group1['SC1'] = group1SC1
Group1['AV1'] = group1AV1
Group1['BC1'] = group1BC1
Group1['CR1'] = group1CR1

Group2 = pd.DataFrame(index=['Top1','Top2', 'Top3', 'Top4', 'Top5', 'Top6', 'Top7', 'Top8', 'Top9', 'Top10'], columns=['AU2', 'Avg2', 'SC2', 'AV2', 'BC2', 'CR2'])
Group2['AU2'] = group2AU2
Group2['Avg2'] = group2Avg2
Group2['SC2'] = group2SC2
Group2['AV2'] = group2AV2
Group2['BC2'] = group2BC2
Group2['CR2'] = group2CR2

Group3 = pd.DataFrame(index=['Top1','Top2', 'Top3', 'Top4', 'Top5', 'Top6', 'Top7', 'Top8', 'Top9', 'Top10'], columns=['AU3', 'Avg3', 'SC3', 'AV3', 'BC3', 'CR3'])
Group3['AU3'] = group3AU3
Group3['Avg3'] = group3Avg3
Group3['SC3'] = group3SC3
Group3['AV3'] = group3AV3
Group3['BC3'] = group3BC3
Group3['CR3'] = group3CR3

print(Group1)

print(Group2)

print(Group3)



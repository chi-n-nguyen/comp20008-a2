max = 0
num = 0
for i in range(len(silhouette_scores)):
    if silhouette_scores[i] > max:
        max = silhouette_scores[i]
        num = i + 2
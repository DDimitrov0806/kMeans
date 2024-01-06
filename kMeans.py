import matplotlib.pyplot as plt
import numpy as np
import re

class KMeans:
    def __init__(self, k, max_iterations=100, random_restarts=10):
        self.k = k  # number of clusters
        self.max_iterations = max_iterations  # maximum number of iterations
        self.random_restarts = random_restarts  # number of times to restart the algorithm

    #initialize centroids using kMeans++ 
    def initialize_centroids(self, points):
        centroids = []
        centroid_index = np.random.randint(len(points))
        centroids.append(points[centroid_index])

        for _ in range(1,self.k):
            distances = []
            for point in points:
                #get the closest centroid to point
                temp = self.calculate_distance_between_point_and_centroids(point, centroids)
                distance = np.min(temp)
                distances.append(distance)
            
            #get the index of the point with max distance
            next_centroid = np.argmax(distances)
            centroids.append(points[next_centroid])

        return centroids

    def assign_points_to_clusters(self, centroids, points):
        clusters = [[] for _ in range(self.k)]
        for point in points:
            #calculate the distance from a point to all the centroids
            distances = self.calculate_distance_between_point_and_centroids(point, centroids)
            #get the closest centroid
            closest_centroid = np.argmin(distances)
            #add the point to the cluster
            clusters[closest_centroid].append(point)
        return clusters

    def calculate_new_centroids(self, clusters):
        #recalculate the centroids to be the average of all the points in the cluster
        centroids = [np.mean(cluster, axis=0).tolist() for cluster in clusters if cluster]  # avoid empty clusters
        return centroids

    def fit(self, points):
        best_score = float('inf')
        best_clusters = None
        best_centroids = None
        
        for _ in range(self.random_restarts):
            centroids = self.initialize_centroids(points)
            for _ in range(self.max_iterations):
                #create clusters
                clusters = self.assign_points_to_clusters(centroids, points)
                #adjust centroids
                new_centroids = self.calculate_new_centroids(clusters)
                
                #if the centroids did not move, finish this step
                if np.all(centroids == new_centroids):
                    break
                
                centroids = new_centroids

            # Evaluate solution
            score = self.evaluate(clusters, centroids)
            if score < best_score:
                best_score = score
                best_clusters = clusters
                best_centroids = centroids

        return best_clusters, best_centroids

    def evaluate(self, clusters, centroids):
        total_distance = 0
        for i,cluster in enumerate(clusters):
            if cluster:  # avoid empty clusters
                for point in cluster:
                    #get the total distance from each point to its centroid
                    total_distance += np.sum(self.calculate_distance_between_point_and_centroids(point, [centroids[i]]))
        return total_distance
    
    def calculate_distance_between_point_and_centroids(self, point, centroids):
        #calculate euclidean distance
        return [np.sqrt((point[0] - centroid[0])**2 +(point[1] - centroid[1])**2) for centroid in centroids]
    
def plot_clusters(clusters, centers):
    for i, center in enumerate(centers):
        cluster = clusters[i]
        x = [point[0] for point in cluster]
        y = [point[1] for point in cluster]
        plt.scatter(x, y, cmap='viridis')
        plt.scatter(center[0], center[1], c='black')

    plt.show()

if '__main__' == __name__:
    file_name = ''
    clusters = 0

    while file_name != 'normal' and file_name != 'unbalance':
        file_name = input()

    while clusters < 1:
        clusters = int(input())

    file_path = './Datasets/{}/{}.txt'.format(file_name, file_name)
    points = []
    with open(file_path, 'r') as file:
        points = file.readlines()

    points = [re.split('[ \t]',line.strip()) for line in points]
    points = [[float(point[0]), float(point[1])] for point in points]

    kMeans = KMeans(clusters)
    clusters, centroids = kMeans.fit(points)

    plot_clusters(clusters, centroids)
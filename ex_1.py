import numpy as np
import sys
import scipy
from scipy import io
from scipy.io import wavfile


def main():
    sample, centroids = sys.argv[1], sys.argv[2]
    fs, y = scipy.io.wavfile.read(sample)
    x = np.array(y.copy())
    points = x
    centroids = np.loadtxt(centroids)
    newCentroids = k_means(centroids, points)
    scipy.io.wavfile.write("compressed.wav", fs, np.array(newCentroids, dtype=np.int16))


def k_means(centroids, points):
    numOfCentroids = len(centroids)
    map = {}
    f = open("output.txt", "w")
    for iter in range(0, 30):
        # reset the classification of the points
        for i in range(0, numOfCentroids):
            l = []
            map[i] = l
        for p in points:
            min = np.inf
            center = 0
            for j in range(0, numOfCentroids):
                distance = np.sqrt(((p[0] - centroids[j][0]) ** 2) + ((p[1] - centroids[j][1]) ** 2))
                if distance < min:
                    min = distance
                    center = j
            # add to array
            map[center].append(p)
        # update the centroids
        isUpdate = False
        for c in range(0, numOfCentroids):
            num = len(map[c])
            if num == 0:
                continue
            sumX = 0
            sumY = 0
            for point in map[c]:
                sumX += point[0]
                sumY += point[1]
            x1 = round(sumX / num)
            y1 = round(sumY / num)
            # check if the point's value updates
            if (x1 != centroids[c][0]) or (y1 != centroids[c][1]):
                centroids[c] = [x1, y1]
                isUpdate = True
        f.write(f"[iter {iter}]:{','.join([str(i) for i in centroids])}\n")
        # all the centroids stay the same
        if not isUpdate:
            break
    f.close()
    return centroids


if __name__ == "__main__":
    main()
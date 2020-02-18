from compareSequence import dtw
import numpy as np

x = [np.random.normal(0,1) for i in range(0,2)]
y = [np.random.normal(0,1) for i in range(0,3)]
print(x)
print(y)

dtwExam = dtw(x, y, 1, 2)
print(dtwExam.dist_ij)
print(dtwExam.dist_mat)
print('------------------------')
print('Coordinate : (1, 2)')
print(dtwExam.dtw_dist)
print(dtwExam.path)

print('------------------------')
dtwExam.set_location(1,1)
print(dtwExam.dtw_dist)
print(dtwExam.path)

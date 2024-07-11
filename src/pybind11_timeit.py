import EigenForPython
import numpy as np
import timeit
a = np.random.random((10,10))
# b = EigenForPython.multiply_matrix_array(a)

print('With Buffer')
print(timeit.repeat('b = EigenForPython.multiply_matrix_array(a)', setup='import EigenForPython; import numpy as np; a = np.random.random((1000,1000))', number=100, repeat=5))
print('Without Buffer')
print(timeit.repeat('b = EigenForPython.multiply_matrix(a)', setup='import EigenForPython; import numpy as np; a = np.random.random((1000,1000))', number=100, repeat=5))
#Q1
import numpy as np
from matplotlib import pyplot as plt

A = np.array([[1, 2, 3], [4, 5, 6]])

# Transpose of A
A_Transpose = np.transpose(A)

A_Square = np.dot(A_Transpose,A)
print("A_Transpose.A Gives us \n",A_Square)


#Q2
import numpy as np
from matplotlib import pyplot as plt
x = np.linspace(-100,100,100)
theta0 = 3
x0 = 1
theta1 = 2
y = theta0 * x0 + theta1 * x
plt.plot(x,y)
plt.title('Question 2')
plt.show()

#Q3
import numpy as np
from matplotlib import pyplot as plt
x = np.linspace(-10,10,100)
theta0 = 4
theta1 = 3
theta2 = 2
y = theta0 * x + theta1 * x + theta2*(x*x)
plt.plot(x,y)
plt.title('Question 3')
plt.show()

#Q4
import numpy as np
from matplotlib import pyplot as plt
x = np.linspace(-100.100,100)
mean = 0
sigma = 15
y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / sigma)**2)
plt.plot(x, y)
plt.title('Gaussian Distribution')
plt.show()

#Q5
import numpy as np
from matplotlib import pyplot as plt
x = np.linspace(-100,100,100)
y = (x*x)
plt.plot(x,y)
plt.title('Question 4')
plt.show()
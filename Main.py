import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create figure and add Axes
figure = plt.figure()
ax = figure.add_subplot(111)
# some sets for ax
ax.set(title='First lab SA',
       xlabel='Среднее значение последовательности из Upper Case',
       ylabel='Самая длинная последовательности из Upper Case',
       #xlim=[-5, 150],
       #ylim=[-50, 2000],
       )
ax.grid()
# read converted dataset
data = pd.read_csv('data.csv')
# take our 2 columns and convert to arrays (55 and 56)
arrX = np.array(data["capital_run_length_average"])
arrY = np.array(data["capital_run_length_longest"])
# add dots to ax
ax.scatter(arrX, arrY, s=9, alpha=0.4)

# create model LinearRegression
model = LinearRegression()
X = pd.DataFrame(
    data.capital_run_length_average)  # Two-dimensional, size-mutable,potentially heterogeneous tabular data
Y = pd.DataFrame(data.capital_run_length_longest)  # size-mutable, potentially heterogeneous tabular data
model.fit(X, Y)  # teach our model
ax.plot(X, model.predict(X), color='red', linewidth=2, label='y = ' + str(model.coef_) +
        'x + ' + str(model.intercept_))  # draw line
ax.legend()
print(np.corrcoef(arrX, arrY))
print('Коэффициент корреляции = ' + str(np.corrcoef(arrX, arrY)[0, 1]))
print(model.score(X, Y))
# Another method for LinearRegression
x = arrX
y = arrY
# lr = np.poly1d(np.polyfit(x, y, 1))  # return regression formula 1x(polynom)
# xlr = np.linspace(0, 150, 150)
# ax.plot(xlr, lr(xlr), color='white', linewidth=1)
# print(lr)

# polinomial 2,3,4
p2 = np.poly1d(np.polyfit(x, y, 2))
print(p2)
xp = np.linspace(0, 1100, 150)
ax.plot(xp, p2(xp), color='green', label=str(p2))


# width and height app
figure.set_figwidth(50)
figure.set_figheight(50)
ax.legend()
plt.show()

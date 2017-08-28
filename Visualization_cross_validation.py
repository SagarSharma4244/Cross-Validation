

from sklearn import linear_model

model = linear_model.LinearRegression()



from sklearn import datasets

boston = datasets.load_boston()
# print(boston)
y = boston.target



from sklearn.model_selection import cross_val_predict

predicted = cross_val_predict(model, boston.data, y, cv=10)



import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()



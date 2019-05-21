
# coding: utf-8

# # Aprendizaje automático, Victor Potes y Ricardo Nuñez

# In[7]:


from sklearn.datasets import load_digits
import pylab as pl
#from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics
import pylab as plt


# In[8]:


digits = load_digits()
numImagenes = len(digits.images) # Numero de imagenes, len es un método que provee el tamaño del arreglo
y = digits.target # el método nos provee las etiquetas de las imágenes en un arreglo
X = digits.images.reshape((numImagenes, -1)) # se reducen las dimensiones


# In[9]:


gnb = GaussianNB()
fit = gnb.fit(X, y)


# In[10]:


y_estimado = fit.predict(X)
print("Reales   :", y[0:25])
print("Estimados:", y_estimado[0:25])


# In[11]:


metrics.accuracy_score(y, y_estimado)


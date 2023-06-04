#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[12]:


#importando el archivo
fifa20 = pd.read_csv(r'C:\Users\Alejandro\Desktop\curso pandas escuela de bayes\players_20.csv')


# In[6]:


#mostrando las primeras 10 filas
fifa20.head(10)


# In[8]:


#mostrando las ultimas 10 filas
fifa20.tail()


# In[9]:


#dimensiones del data set
fifa20.shape


# In[10]:


#nombres de las columnas
fifa20.columns.values


# In[11]:


#datos estadisticos basicos del data set
fifa20.describe()


# In[12]:


#tipo de dato de cada variable
fifa20.dtypes


# In[13]:


#se selectionan del dataset algunas columnas que nesecitamos usar, y la asignamos a una variable
fifa20_1 = fifa20[['short_name', 'age', 'height_cm', 'weight_kg', 'nationality', 'club', 'overall', 'potential', 'value_eur',
                  'wage_eur', 'player_positions', 'power_jumping', 'power_long_shots']]


# In[15]:


fifa20_1.head()


# In[17]:


type(fifa20_1)


# In[29]:


#me muestra los datos del equipo que le estpy pidiendo de acunerdo al nuevo data set que hice
realmadrid = fifa20_1[fifa20_1['club'] == 'Real Madrid']
print(realmadrid.shape)


# In[31]:


#me muestra los datos del equipo que le estpy pidiendo de acunerdo al nuevo data set que hice
barcelona = fifa20_1[fifa20_1['club'] == 'FC Barcelona']
print(barcelona.shape)


# In[32]:


realmadrid.head()


# In[33]:


#que club tiene los mejores jugadores real vs barca
plt.figure()

plt.subplot(121)
plt.boxplot(realmadrid['overall'])
plt.ylabel('Overall')
plt.title('Real Madrid')
plt.grid(True)

plt.subplot(122)
plt.boxplot(barcelona['overall'])
plt.title('Barcelona')
plt.grid(True)


# In[34]:


print(realmadrid['overall'].describe())


# In[35]:


print(barcelona['overall'].describe())


# In[36]:


#que club tiene los jugadors mejor valorados
realmadrid['value_eur'].describe()


# In[38]:


plt.hist(realmadrid['value_eur'])
plt.xlabel('Valor en 10x8 de millones de Euros')
plt.ylabel('Frecuencia')
plt.title('Valor en 10x8 de millones de Euros - Real Madrid')
plt.grid(True)


# In[39]:


barcelona['value_eur'].describe()


# In[40]:


plt.hist(barcelona['value_eur'])
plt.xlabel('Valor en 10x8 de millones de Euros')
plt.ylabel('Frecuencia')
plt.title('Valor en 10x8 de millones de Euros - Barcelona')
plt.grid(True)


# In[41]:


mostvalue_rm = realmadrid[realmadrid['value_eur'] > 50000000]
mostvalue_rm


# In[42]:


mostvalue_bz = barcelona[barcelona['value_eur'] > 50000000]
mostvalue_bz


# In[45]:


#relacion de edades y valoraciones en 10x8 de millones de euros
fifa20_1.plot(kind = 'scatter', x = 'age', y = 'value_eur')


# In[46]:


#jugadores mas valiosos
fifa20_1.describe()


# In[47]:


mostvalue_world = fifa20_1[fifa20_1['value_eur'] > 80000000]
print(fifa20_1.shape)
print(mostvalue_world.shape)
mostvalue_world


# In[48]:


#el club con los mejores salarios entre barca y real
realmadrid['wage_eur'].describe()


# In[49]:


barcelona['wage_eur'].describe()


# In[50]:


#jugadores con salarios mayores a 300k eur
best_wage_rm = realmadrid[realmadrid['wage_eur'] > 300000]
best_wage_rm


# In[51]:


best_wage_bz = barcelona[barcelona['wage_eur'] > 300000]
best_wage_bz


# In[52]:


#que jugador tiene los mejores salarios de fifa 20
fifa20_1['wage_eur'].describe()


# In[53]:


#que jugador tiene le mjro salario
best_wage_world = fifa20_1[fifa20_1['wage_eur'] > 250000]
best_wage_world


# In[58]:


#potencia en tiros
plt.hist(fifa20_1['power_long_shots'])
plt.xlabel('Potencia en tiros largos')
plt.ylabel('Frecuencia')
plt.title('Potencia en tiros largos')
plt.grid(True)


# In[59]:


fifa20_1['power_long_shots'].describe()


# In[60]:


best_powershots = fifa20_1[fifa20_1['power_long_shots'] > 89]
best_powershots


# In[61]:


#potencia en salto
plt.hist(fifa20_1['power_jumping'])
plt.xlabel('Potencia en salto')
plt.ylabel('Frecuencia')
plt.title('Potencia en salto')
plt.grid()


# In[62]:


fifa20_1['power_jumping'].describe()


# In[63]:


best_powerjumping = fifa20_1[fifa20_1['power_jumping'] > 92]
best_powerjumping


# In[64]:


#jugadores con mayor edad
fifa20_1['age'].describe()


# In[66]:


age_old = fifa20_1[fifa20_1['age'] > 40]
age_old


# In[69]:


fifa20.columns.values


# In[70]:


#relacion de variables
fifa20_2 = fifa20[['age', 'height_cm', 'weight_kg', 'overall', 'potential', 'value_eur',
                  'wage_eur', 'power_jumping', 'power_long_shots', 'skill_moves', 
                   'shooting', 'passing', 'dribbling', 'defending', 'physic', 
                  'movement_acceleration', 'movement_agility', 'mentality_vision']]


# In[71]:


#coorrelacion
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = fifa20_2.corr()
sns.heatmap(corr)


# In[72]:


#existe una coorralcion alta con estas dos variables
fifa20_2.plot(kind = 'scatter', x = 'mentality_vision',  y = 'passing', color = 'orange')


# In[73]:


#coorealcion mayor
fifa20_2.plot(kind = 'scatter', x = 'mentality_vision',  y = 'dribbling', color = 'blue')


# In[74]:


fifa20_2.plot(kind = 'scatter', x = 'height_cm',  y = 'movement_agility', color = 'purple')


# In[76]:


#creacion de series y dataframes
obj = pd.Series([4,7, -5,3])
obj


# In[77]:


obj.values


# In[78]:


obj.index


# In[80]:


#se le asignan los indices que queramos
obj2 = pd.Series([4,7,-5,3], index = ['d','b', 'a', 'c'])
obj2


# In[81]:


obj2['a']


# In[82]:


obj2[['a','b']]


# In[83]:


obj2[obj2 > 0]


# In[84]:


obj2 * 2


# In[85]:


sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utha': 5000}
obj3 = pd.Series(sdata)
obj3


# In[86]:


states = {'California', 'Ohio', 'Oregon', 'Texas'}
obj4 = pd.Series(sdata,index = states)
obj4


# In[87]:


#trabajndo con valores faltantes
pd.isnull(obj4)


# In[88]:


pd.notnull(obj4)


# In[89]:


obj4.isnull()


# In[90]:


obj3 + obj4


# In[91]:


obj4.name = 'poblacion'


# In[93]:


obj4.index.name = 'estado'
obj4


# In[1]:


#dataframes
import numpy as np


# In[4]:


#se crea el dataframe
data = {'state':['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
       'year': [2000,2001,2002,2001,2002,2003],
       'pop':[1.5,1.7,3.6,2.4,2.9,3.2]}
frame = pd.DataFrame(data)


# In[5]:


frame


# In[6]:


frame.head()


# In[7]:


#para intercambiar las columnas
pd.DataFrame(data, columns = ['year', 'state', 'pop'])


# In[8]:


#se crea un nuevo data frame con una columna nueva
frame2 = pd.DataFrame(data, columns = ['year', 'state', 'pop', 'debt'], index = ['one', 'two', 'three', 'four', 'five', 'six'])


# In[9]:


frame2


# In[10]:


frame2.columns


# In[11]:


#ver la columna que queramos
frame['state']


# In[14]:


frame2['debt'] = 16.5
frame2


# In[15]:


#se genrean nuemero de 0 al 5 para la columna debt
frame2['debt'] = np.arange(6)


# In[19]:


#importando el archivo
fifa20 = pd.read_csv(r'C:\Users\Alejandro\Desktop\curso pandas escuela de bayes\players_20.csv')
                     


# In[18]:


import os


# In[20]:


fifa20.head(2)


# In[2]:


dfhr = pd.read_csv(r'C:\Users\Alejandro\Desktop\curso pandas escuela de bayes\recursos_humanos.csv')


# In[ ]:





# In[24]:


dfhr.shape


# In[25]:


dfhr.dtypes


# In[26]:


dfhr.head()


# In[28]:


dfhr.tail()


# In[32]:


dfhr1 = dfhr[['Employee_Name', 'Salary', 'Position', 'State', 'Sex', 'DOB', 'MaritalDesc', 'CitizenDesc', 'Department', 'PerformanceScore', 'EngagementSurvey', 'EmpSatisfaction']]


# In[33]:





# In[34]:


dfhr1.head()


# In[41]:


dfhr1.describe()


# In[44]:


titanic = pd.read_csv(r'C:\Users\Alejandro\Desktop\curso pandas escuela de bayes\titanic.csv')


# In[45]:


titanic.head(50)


# In[46]:


titanic.shape


# In[47]:


#elimina las filas que le falten todos lo valores
titanic1 = titanic.dropna(axis = 0, how = 'all')


# In[48]:


titanic.shape


# In[50]:


#elimina las filas que le falte algun valor
titanic2 = titanic.dropna(axis = 0, how = 'any')


# In[51]:


titanic2.shape


# In[55]:


#rellenar nan cuando sean pocos
#en la columna edad faltan 177 pasajeros sin la edad
pd.isnull(titanic['Age']).values.ravel().sum()


# In[56]:


#se rellenan las edades faltantes con el promedio de la tabla
col = titanic['Age'].fillna(titanic['Age'].mean())


# In[61]:


col.head(50)


# In[64]:


titanic1.head()


# In[69]:


#one hot encoding
dfhr.head()


# In[70]:


dfhr1 = dfhr[['Employee_Name', 'Salary', 'Position', 'State', 'Sex', 'DOB', 'MaritalDesc',
             'CitizenDesc', 'Department', 'PerformanceScore', 'EngagementSurvey', 'EmpSatisfaction']]


# In[71]:


dfhr1.shape


# In[73]:


dfhr1.describe()


# In[74]:


dfhr1.head(5)


# In[75]:


#creamos un one hot encodign
dummy_sex = pd.get_dummies(dfhr1['Sex'], prefix = 'sex')


# In[76]:


dummy_sex.head()


# In[77]:


#eliminamos la columna Sex
dfhr1 = dfhr1.drop(['Sex'], axis = 1)


# In[78]:


#verificamos si se elimno la columna
dfhr1.shape


# In[80]:


#creamos un nuevo dataset yconcatenamos el dataset con el dumy que crqsmos
dfhr2 = pd.concat([dfhr1, dummy_sex], axis=1)


# In[81]:


dfhr2.shape


# In[82]:


dfhr2.head()


# In[83]:


#automatizar one hot encoding
def createdummies(df, varname):
    dummy = pd.get_dummies(df[varname], prefix = varname)
    df = df.drop(varname, axis = 1)
    df = pd.concat([df, dummy], axis = 1)
    return df


# In[84]:


createdummies(dfhr2, 'CitizenDesc')


# In[3]:


df = dfhr[['Employee_Name', 'Salary', 'PerformanceScore', 'EngagementSurvey']]


# In[6]:


df = df.head(6)
df


# In[8]:


df1 = dfhr[['Employee_Name', 'Salary', 'PerformanceScore', 'EngagementSurvey']]
df1


# In[9]:


df1 = df1.iloc[7:12]
df1


# In[10]:


pd.concat([df, df1])


# In[11]:


df5 = pd.concat([df, df1])


# In[13]:


df6 = df5[['Employee_Name']]
df6


# In[15]:


df7 = df5[['Salary', 'PerformanceScore', 'EngagementSurvey']]
df7


# In[16]:


pd.concat([df6, df7], axis = 1)


# In[19]:


#para ordenar de menor a mayor
fifa20.head()


# In[20]:


fifa20_1 = fifa20[['short_name','age', 'height_cm', 'weight_kg', 'nationality', 'club',
                  'overall', 'potential', 'value_eur', 'wage_eur', 'player_positions', 'power_jumping', 'power_long_shots']]


# In[21]:


fifa20_1.head()


# In[22]:


#ordenr de menor a mayor la columna q le estoy diciendo
fifa20_1.sort_values('overall').head()


# In[23]:


#ordenr de mayor a menor la columna q le estoy diciendo
fifa20_1.sort_values('overall', ascending = False).head()


# In[24]:


rm = fifa20_1[fifa20_1['club'] == 'Real Madrid']
print(rm.shape)


# In[25]:


rm.head()


# In[26]:


rm.sort_values('overall', ascending = False).head()


# In[27]:


rm.sort_values('value_eur', ascending = False).head()


# In[28]:


#cambiar  nombre de las columnas
rm.rename(columns = {'age':'edad'}).head()


# In[29]:


rm.rename(columns = {'player_positions':'posicion'}).head()


# In[31]:


real = rm.drop(columns = ['short_name', 'age'])


# In[32]:


real.head()


# In[33]:


#mostra r filas que cumplan un criterio
rm[rm.overall > 88]


# In[34]:


#nuestra los juegadores mayores de 30 años
rm[rm.age > 30]


# In[35]:


#nuestra los juegadores que pesan mas de 83 kg

rm[rm.weight_kg > 83]


# In[36]:


#seleccion aleatoria(fraccion)
rm.shape


# In[37]:


rm.head()


# In[38]:


rm.sample(frac=0.1)#fraccion en porcentaje de filas, de las 33 seleciona el 10%


# In[39]:


#seleccion aleatoira de filas en cantidad, mas no en porcentaje
rm.sample(n=5)


# In[2]:


df = pd.read_csv(r'C:\Users\Alejandro\Desktop\curso pandas escuela de bayes\marketing.csv')


# In[41]:


df.head()


# In[42]:


df.shape


# In[43]:


#seleccion de filas de la 10 a la 20. La 10 inclsiva y la 20 no se incluye
df.iloc[10:20]


# In[44]:


df.columns.values


# In[45]:


#se utiliza el siguiente codigo para encontrar los mayores 5 valores de la columna
df.nlargest(5, 'MntFruits')


# In[47]:


#se utiliza el siguiente codigo para encontrar los menores 5 valores de la columna
df.nsmallest(5, 'MntWines')


# In[48]:


#con esot se encuentra los valores mas altos en la columna que le asignemos 
df.nlargest(5, 'Recency' )


# In[49]:


#con esot se encuentra los valores mas bajos en la columna que le asignemos 
df.nsmallest(5, 'Recency' )


# In[50]:


# se seleccioan las columnas 
df[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts']].head()


# In[51]:


#se selecciona una sola columna
df['MntWines']


# In[52]:


#se selecciona una sola columna, otra forma
df.MntWines


# In[5]:


#selcccion de columnas q inicien con Mn
df.filter(regex = '^Mn').head()


# In[4]:


df.Teenhome


# In[6]:


#selcccion de columnas q finaliccen con home
df.filter(regex = 'home$').head()


# In[7]:


#se seleccionan las columnas desde educationbasic hasata phd invclyendise, los : sugnifican q todas las filas
df.loc[:, 'education_Basic': 'education_PhD'].head()


# In[9]:


#aqui se selcciona las columnas 1,2,5
df.iloc[:,[1,2,5]].head()


# In[10]:


#numero de personas con education_Basic 0 no, 1 si
df['education_Basic'].value_counts()


# In[11]:


df.columns.values


# In[12]:


df['marital_Married'].value_counts()


# In[13]:


#se crean 2 dataset con las dfertes campañas
df1 = df[df.AcceptedCmp1 == 1]
df2 = df[df.AcceptedCmp2 == 0]


# In[15]:


#que nos quiere decir, que de los q aceptaron la campaña cumplieron y respondieron
print(df1['Complain'].value_counts())
print(df1['Response'].value_counts())


# In[16]:


print(df2['Complain'].value_counts())
print(df2['Response'].value_counts())


# In[17]:


df1.describe()


# In[18]:


df2.describe()


# In[6]:


'la tasa de conversion en la primera cmapaña es de ' + str((df1.shape[0] / (df2.shape[0] + df1.shape[0]) * 100 + '%'))


# In[3]:


#comparacion de los clientes q aceptaron la oferta en la primera campaña vs ultima campaña
firstc = df[df['AcceptedCmp1'] == 1]
lastc = df[df['Response'] == 1]
print(firstc.shape, lastc.shape )


# In[4]:


#tasa de conversion
tc1 = ((firstc.shape[0] / df.shape[0]) * 100)


# In[5]:


print('la tasa de conversion e la primera campaña fue de ' + str(tc1) + '%')


# In[7]:


#promedio de las personas que estuviern en la primera camapña
firstc.mean()


# In[8]:


#quienes son los mas jovenes
print(firstc.mean()[1], lastc.mean()[1])


# In[10]:


#numero de dias desde la ultima compra
print(firstc.mean()[3], lastc.mean()[3])


# In[11]:


#cantidad de vino q conusmen
print(firstc.mean()[4], lastc.mean()[4])


# In[14]:


fifa20_1.shape


# In[15]:


fifa20_1.columns


# In[19]:


#jugadores del barca con menos de 20 años
barca_final = fifa20_1[(fifa20_1['club'] == 'FC Barcelona') & (fifa20_1['age'] < 20)]
barca_final


# In[22]:


#merge
#se crea un dataframe
staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'}, 
                        {'Name': 'Sally', 'Role': 'Course liasion'},
                        {'Name': 'James', 'Role': 'Grader'}])


# In[23]:


#asignamos el index por la columna Name
staff_df = staff_df.set_index('Name')


# In[24]:


student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                          {'Name': 'Mike', 'School': 'Law'},
                          {'Name': 'Sally', 'School': 'Engineering'}])


# In[26]:


student_df = student_df.set_index('Name')


# In[27]:


staff_df.head()


# In[28]:


student_df.head()


# In[29]:


#outer
#se obtiene un dataset con todos los registros de los dos dataset
pd.merge(staff_df, student_df, how = 'outer', left_index = True, right_index = True)
#how = outer significa que el tipo de union q se va aaplicar
#left_index y right_index ambos True significa que es el indice que se va tomar en la union de lso dos dataset


# In[30]:


#inner
#aqui obtenemos un data set con los registros en comun de los dos dataset
pd.merge(staff_df, student_df, how = 'inner', left_index = True, right_index = True)


# In[31]:


#aqui obtenemos un dataset con los registros del dataset izquierdo y con la infoormacion
# de las columnas correspondientes del dataset derecho
pd.merge(staff_df, student_df, how = 'left', left_index = True, right_index = True)


# In[32]:


#
pd.merge(staff_df, student_df, how = 'right', left_index = True, right_index = True)


# In[33]:


#se eliminan los indices asignados previamente en los dos dataset
staff_df = staff_df.reset_index()
student_df = student_df.reset_index()


# In[34]:


#aqui obtnemos un dataset con los registros de l dataset derechos y 
# con la informacion de las columnas correspondeintes del dataset izquierdo
pd.merge(staff_df, student_df, how = 'right', on = 'Name')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





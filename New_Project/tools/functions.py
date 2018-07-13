def drop_lines(df, column, valor):
    if valor=='NaN':
        indexes = df[df[column].isnull()].index
    else:
        indexes = df[df[column] == valor].index
    df = df.drop(indexes)
    return (df)

def letters_to_num(obj, L, N, column):
    obj[column] = obj[column].replace(L, N)
    return obj

def Do_resamples(df, col, typ):
    p = get_proportions(df, col)
    df, r = resample_NaN_proportion(df, col, p, typ)
    while(r != 0):
        df, r = resample_NaN_proportion(df, col, p, typ)

def get_proportions(df, col):
    """This function get the proportion of each value found in the DataFramea into a list"""
    p = []
    for i, j in enumerate(df[col].value_counts().sort_index().values, 1):
        p.append((j)/df[col].value_counts().sort_index().values.sum())
    T = df[col].isnull().sum()
    for i in range(len(p)):
        p[i] = round(round(p[i],8)*T)
    return(p)

def resample_NaN_proportion(df, col, p, typ):
    """This function put a value in the position of the NaN values. And Returns the dataframe modified and the amount of remaining NaN values"""
    import numpy as np
    indexes = df[df[col].isnull()].index
    print('cant null antes:', df[col].isnull().sum())
    c = 0
    for i in p:
        ite = indexes[c:c+i]
        cl = df[col].value_counts().sort_index().index[p.index(i)]
        for j in ite:
            df.loc[j, col] = typ(cl)
        c += i
    print('cant null despues:', df[col].isnull().sum())
    r = df[col].isnull().sum()
    return(df, r)

def shape_cols(obj):
    print("Columns found: ", obj.describe().shape[1])

def obj_2list(*objs):
    l = []
    for i in objs:
        r = []
        for j in i.columns:
            r.append(j)
        l.append(r)
    return l

def obj_2set(*objs):
    l = []
    for i in objs:
        r = set()
        for j in i.columns:
            r.add(j)
        l.append(r)
    return l

def create_infotable(*obj):
    l = []
    for i in obj:
        r = []
        r.append(i.shape[0])
        r.append(i.shape[1])
        r.append(i.size)
        l.append(r)
    return l

#Obtiene los pesos en MB (MegaBytes) de los archivos, y los devuelve en una lista junto con el año que indica de que archivo era
def get_weights(l):
    r = []
    for i in l:
        t = []
        year = i.split("..")[1].split("/")[-1].split("-")[1]
        t.append(year[:-1]+"_"+year[-1])
        t.append(i.split("..")[0].split("\t")[0][0:-1])
        r.append(t)
    return r


#Esta funcion "une" dos filas del DataFrame Pasado como parametro
#Exactamente lo que hace es sumar los valores de las dos filas, valor por valor, como tengo 1s y 0s me sirve una suma simple
#Ademas de esto le cambia el valor al nombre de la fila resultante y le hace drop a la segunda fila.
def merge_rows(df, ind1, ind2):
    newname = df.loc[ind1][0]
    df.loc[ind1] = df.loc[ind1] + df.loc[ind2]
    df = df.drop([ind2])
    oldname = df.loc[ind1][0]
    df = df.replace(to_replace=oldname, value=newname)
    return df

#Esta funcion es para ver que tantos nulls/NAN hay
def missing_data(obj):
    print("Elementos null/NaN", obj.isnull().sum())
    print("__________________")
    obj2=obj.dropna()
    print("Elementos diferentes de null/NaN:", obj2[obj2.isnull()!=True].count())
    print("__________________")
    print(obj2.shape)
    
    
    
#Funcion para limpiar los TARGETS de algunos archivos
def clean_target(df, c):
    """Replace the ',' for '.' and transform them into numbers"""
    for i in df[c].value_counts().index:
        if ',' in i:
            df[c] = df[c].replace(i, i.replace(',', '.'))
    
    for i in df[c].value_counts().index:
        if type(i) == str:
            df[c] = df[c].replace(i, float(i))

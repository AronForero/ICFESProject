def letters_to_num(obj):
    L = ["N", "S", "A", "B", "C", "D", "E", "F", "G", "H"]
    N = [-1, -2, -10, -11, -12, -13, -14, -15, -16, -17]
    x = obj.replace(L, N)
    return x

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

#Esta funcion es para ver que tantos nulls/NAN hay
def missing_data(obj):
    print(obj[obj.isnull()!=True].count())
    print("__________________")
    flights2=obj.dropna()
    print(flights2[flights2.isnull()!=True].count())
    print("__________________")
    print(flights2.shape)

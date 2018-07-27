def label_encoder(df):
    assert sum(df.dtypes == 'object') >= 1, 'dataframe does not have any column of type object'
    #se saca la lista de columnas de tipo object
    df1 = df.copy()
    list_columns_object = df1.dtypes[df1.dtypes == 'object'].index
    for index in list_columns_object:
        #se cambian los tipos a category para poder aplicar el .cat.codes de pandas
        df1.loc[:,index] = df1.loc[:,index].astype('category').cat.codes
        
    return df1
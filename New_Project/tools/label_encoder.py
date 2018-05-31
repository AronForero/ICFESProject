def label_encoder(df):
    assert sum(df.dtypes == 'object') >= 1, 'dataframe not have a values of type object'
    #se saca la lista de columnas de tipo object
    list_columns_object = df.dtypes[df.dtypes == 'object'].index
    for index in list_columns_object:
        #se cambian los tipos a category para poder aplicar el .cat.codes de pandas
        df.loc[:,index] = df.loc[:,index].astype('category').cat.codes
        
    return df
def Prepare_data_2014(data, c):
    """Returns the data ready to be used in a prediction.
    
    -data: full dataset
    -c: column/subject to predict
    
    returns:
        -X2: New Predictive variables powered to 2
        -LR_Data: Old Predictive variables transformed by PCA of 35 components. This will be used for the Linear Regression Object
        -DT_Data: Old Predictive variables transformed by PCA of 33 components. This will be used for the Decision Tree Object
        -Y: Dataframe with the chosen target and the others"""
    data = data.sort_values(by=c) 
    y_list = ['DECIL_LECTURA_CRITICA', 'PUNT_LECTURA_CRITICA', 'DECIL_MATEMATICAS', 'PUNT_MATEMATICAS', 'DECIL_C_NATURALES',
              'PUNT_C_NATURALES', 'DECIL_SOCIALES_CIUDADANAS', 'PUNT_SOCIALES_CIUDADANAS', 'DECIL_INGLES', 'DESEMP_INGLES',
              'PUNT_INGLES', 'DECIL_RAZONA_CUANT', 'PUNT_RAZONA_CUANT', 'DECIL_COMP_CIUDADANA', 'PUNT_COMP_CIUDADANA', 
              'PUNT_GLOBAL', 'ESTU_PUESTO']
    new_y_list = ['PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES',
             'PUNT_RAZONA_CUANT', 'PUNT_COMP_CIUDADANA', 'PUNT_GLOBAL']
    X_list = data.columns.difference(y_list)
    
    Y = data.filter(items = new_y_list)
    X_data = data.filter(items = X_list)
    
    DT_Data = X_data
    RF_Data = X_data
    
    ward = Prepare_RFR_data_2014(data)
    if ward == 1:
        New_x_list = ['COLE_AREA_UBICACION', 'COLE_BILINGUE', 'COLE_CALENDARIO', 'COLE_CARACTER', 'COLE_COD_ICFES',
                      'COLE_GENERO', 'COLE_JORNADA', 'COLE_NATURALEZA', 'ESTU_COD_OTRO_PAIS_PLANTEL', 'ESTU_COD_PLANTEL',
                      'ESTU_COD_RESIDE_DEPTO', 'ESTU_DEPTO_PRESENTACION', 'ESTU_EDAD', 'ESTU_ESTUDIANTE', 'ESTU_ETNIA',
                      'ESTU_GENERO', 'ESTU_IES_COD_DESEADA', 'ESTU_IES_MPIO_DESEADA', 'ESTU_MCPIO_PRESENTACION',
                      'ESTU_NACIMIENTO_DIA', 'ESTU_NACIMIENTO_MES', 'ESTU_PAIS_RESIDE', 'ESTU_PRESENTO_ANTECEDENTES',
                      'ESTU_PRESENTO_EXPECTATIVAS', 'ESTU_PRIVADO_LIBERTAD', 'ESTU_TIPO_CARRERA_DESEADA', 'ESTU_TRABAJA',
                      'ESTU_VECES_ESTADO', 'ESTU_VECES_ESTADO_ESTUDIANTIL', 'ESTU_ZONA_RESIDE', 'FAMI_AUTOMOVIL',
                      'FAMI_CELULAR', 'FAMI_CUARTOS_HOGAR', 'FAMI_DVD', 'FAMI_EDUCA_PADRE', 'FAMI_HORNO',
                      'FAMI_INGRESO_FMILIAR_MENSUAL', 'FAMI_INTERNET', 'FAMI_LAVADORA','FAMI_MICROONDAS', 'FAMI_NEVERA',
                      'FAMI_NIVEL_SISBEN', 'FAMI_OCUPA_MADRE', 'FAMI_OCUPA_PADRE','FAMI_PERSONAS_HOGAR', 'FAMI_PISOSHOGAR',
                      'FAMI_SERVICIO_TELEVISION', 'FAMI_TELEFONO_FIJO']
        X = data.filter(items=New_x_list)
        
        LR_Data = X**2
    else:
        X = 0
        LR_Data = X
    
    return(LR_Data, DT_Data, RF_Data, Y)

def Prepare_RFR_data_2014(data):
    """This function checks if all of the columns contained in the New X data are in the dataframe passed.
    
    Returns:
        1 if all the columns are
        0 if aren't 
    """
    ward = 0
    X_list = ['COLE_AREA_UBICACION', 'COLE_BILINGUE', 'COLE_CALENDARIO', 'COLE_CARACTER', 'COLE_COD_ICFES', 'COLE_GENERO',
              'COLE_JORNADA', 'COLE_NATURALEZA', 'ESTU_COD_OTRO_PAIS_PLANTEL', 'ESTU_COD_PLANTEL', 'ESTU_COD_RESIDE_DEPTO',
              'ESTU_DEPTO_PRESENTACION', 'ESTU_EDAD', 'ESTU_ESTUDIANTE', 'ESTU_ETNIA', 'ESTU_GENERO', 
              'ESTU_IES_COD_DESEADA', 'ESTU_IES_MPIO_DESEADA', 'ESTU_MCPIO_PRESENTACION', 'ESTU_NACIMIENTO_DIA',
              'ESTU_NACIMIENTO_MES', 'ESTU_PAIS_RESIDE', 'ESTU_PRESENTO_ANTECEDENTES', 'ESTU_PRESENTO_EXPECTATIVAS',
              'ESTU_PRIVADO_LIBERTAD', 'ESTU_TIPO_CARRERA_DESEADA', 'ESTU_TRABAJA', 'ESTU_VECES_ESTADO',
              'ESTU_VECES_ESTADO_ESTUDIANTIL', 'ESTU_ZONA_RESIDE', 'FAMI_AUTOMOVIL', 'FAMI_CELULAR', 'FAMI_CUARTOS_HOGAR',
              'FAMI_DVD', 'FAMI_EDUCA_PADRE', 'FAMI_HORNO', 'FAMI_INGRESO_FMILIAR_MENSUAL', 'FAMI_INTERNET', 'FAMI_LAVADORA',
              'FAMI_MICROONDAS', 'FAMI_NEVERA', 'FAMI_NIVEL_SISBEN', 'FAMI_OCUPA_MADRE', 'FAMI_OCUPA_PADRE',
              'FAMI_PERSONAS_HOGAR', 'FAMI_PISOSHOGAR', 'FAMI_SERVICIO_TELEVISION', 'FAMI_TELEFONO_FIJO']
    y_list = ['DECIL_LECTURA_CRITICA', 'PUNT_LECTURA_CRITICA', 'DECIL_MATEMATICAS', 'PUNT_MATEMATICAS', 'DECIL_C_NATURALES',
              'PUNT_C_NATURALES', 'DECIL_SOCIALES_CIUDADANAS', 'PUNT_SOCIALES_CIUDADANAS', 'DECIL_INGLES', 'DESEMP_INGLES',
              'PUNT_INGLES', 'DECIL_RAZONA_CUANT', 'PUNT_RAZONA_CUANT', 'DECIL_COMP_CIUDADANA', 'PUNT_COMP_CIUDADANA', 
              'PUNT_GLOBAL', 'ESTU_PUESTO']
    diff = data.columns.difference(y_list)
    for i in X_list:
        if i in diff:
            ward+=1
    
    if ward == 48:
        return(1)
    else:
        return(0)


def Get_Scores_2014(RF_Data, LR_Data, DT_Data, Y, c):
    """Train the three models (LinearRegression, DecisionTreeRegressor, RandomForestRegressor), and 
    test them. All of this with a cross_val_score object
    
    -X2: New Predictive variables powered to 2
    -LR_Data: Old Predictive variables transformed by PCA of 35 components. This will be used for the Linear Regression Object
    -DT_Data: Old Predictive variables transformed by PCA of 33 components. This will be used for the Decision Tree Object
    -Y: Dataframe with the chosen target and the others
    -c: Chosen column/subject
    
    Returns:
        -Scores(MEAN ABSOLUTE ERROR) of the trained models"""
    
    if type(LR_Data) != pd.core.frame.DataFrame:
        LR_scores = [-1,-1,-1,-1,-1]
    else:
        cv = ShuffleSplit(n = LR_Data.shape[0], n_iter=5, test_size=0.2)
        LR = LinearRegression(n_jobs=4)
        LR_scores = -cross_val_score(LR, LR_Data, Y[c], scoring='mean_absolute_error', cv = cv, n_jobs=4)
    
    if type(DT_Data) != pd.core.frame.DataFrame:
        DT_scores = [-1,-1,-1,-1,-1]
    else:
        cv = ShuffleSplit(n = DT_Data.shape[0], n_iter=5, test_size=0.2)
        DT = DecisionTreeRegressor(max_depth=10)
        DT_scores = -cross_val_score(DT, DT_Data, Y[c], scoring='mean_absolute_error', cv = cv, n_jobs=4)
    
    if type(RF_Data) != pd.core.frame.DataFrame:
        RF_scores = [-1,-1,-1,-1,-1]
    else:
        cv = ShuffleSplit(n = RF_Data.shape[0], n_iter=5, test_size=0.2)
        RF = RandomForestRegressor(max_depth=11, n_jobs=4)
        RF_scores = -cross_val_score(RF, RF_Data, Y[c], scoring='mean_absolute_error', cv = cv, n_jobs=4)
        
    return(LR_scores, DT_scores, RF_scores)

def Predict_all_Subjects_2014(dataset):
    """Uses the Automation functions to train and test models for each Subject in the given Dataset.
    dataset: Dataset to be used for the models
    
    Returns:
        -One Dataset with all the Scores indexed by subject"""
    
    new_y_list = ['PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES',
                  'PUNT_RAZONA_CUANT', 'PUNT_COMP_CIUDADANA', 'PUNT_GLOBAL']
    Y_slice = dataset.filter(items=new_y_list)
    
    LR_List = []
    DT_List = []
    RF_List = []
    idx_List = []
    for i in Y_slice.columns:
        LR_Data, DT_Data, RF_Data, Y = Prepare_data_2014(dataset, i)
        LR_scores, DT_scores, RF_scores = Get_Scores_2014(RF_Data, LR_Data, DT_Data, Y, i)
        Show_Score(LR_scores, DT_scores, RF_scores, i)
        idx_List.append(i)
        LR_List.append(np.mean(LR_scores))
        DT_List.append(np.mean(DT_scores))
        RF_List.append(np.mean(RF_scores))
    Scores_List = {'Subject': idx_List, 'LR': LR_List, 'DTR': DT_List, 'RFR': RF_List}
    Scores_DF = pd.DataFrame(Scores_List)
    return(Scores_DF)

#############################################################################################
############################################################################################
def Test_with_2sets_2014(dataset1, dataset2, c):
    """Train three models with the first Dataset, and Test the models with the second Dataset,
    -Shows the Mean Absolute Error of each model.
    
    -dataset1: First dataset, will be used to train the models
    -dataset2: Second dataset, will be used to test the models
    -c: Column/Subject to predict
    
    Returns:
        -The trained Models
        -The Score of each Model"""
    
   
    LR_Data, DT_Data, RF_Data, Y = Prepare_data_2014(dataset1, c)
    LR_Data_2, DT_Data_2, RF_Data_2, Y_2 = Prepare_data_2014(dataset2, c)
    
    
    ##################TRAINING AND TESTING ######################
    print('Scores for the Columns/Subject:', c)
    if type(LR_Data) != pd.core.frame.DataFrame:
        LR = 0
        LR_Score = -1
        print('DATA ERROR... Linear Regression: No se pudo entrenar el modelo.')
    else:
        LR = LinearRegression(n_jobs=4);
        LR.fit(LR_Data, Y[c]);
        if type(LR_Data_2) != pd.core.frame.DataFrame:
            print('DATA ERROR... Linear Regression: Se pudo entrenar pero no validar.')
            LR_Score = -1
        else:
            try:
                LR_Score = MAD(Y_2[c], LR.predict(LR_Data_2))
                print('Score LR:', LR_Score)
            except
    
    if type(DT_Data) != pd.core.frame.DataFrame:
        DT = 0
        DT_Score = -1
        print('DATA ERROR... Decision Tree: No se pudo entrenar el modelo.')
    else:
        DT = DecisionTreeRegressor(max_depth=7);
        DT.fit(DT_Data, Y[c]);
        if type(DT_Data_2) != pd.core.frame.DataFrame:
            print('DATA ERROR... Decision Tree: Se pudo entrenar pero no validar.')
            DT_Score = -1
        else:
            DT_Score = MAD(Y_2[c], DT.predict(DT_Data_2))
            print('Score DT:', DT_Score)
    
    if type(RF_Data) != pd.core.frame.DataFrame:
        RF = 0
        RF_Score = -1
        print('DATA ERROR... Random Forest: No se pudo entrenar el modelo.')
    else:
        RF = RandomForestRegressor(max_depth=10, n_jobs=4);
        RF.fit(RF_Data, Y[c]);
        if type(RF_Data_2) != pd.core.frame.DataFrame:
            print('DATA ERROR... Random Forest: Se pudo entrenar pero no validar.')
            RF_Score = -1
        else:
            RF_Score = MAD(Y_2[c], RF.predict(RF_Data_2))
            print('Score RF:', RF_Score)
            
    print()
    
    return(LR, LR_Score, DT, DT_Score, RF, RF_Score)

def Test_all_Subjects_2014(dataset1, dataset2):
    """Uses the Test_with_2sets function to do the corresponding predictions for each Subject.
    
    Shows the Prediction of the Subject for each model. 
    
    dataset1: First dataset, will be used to predict
    dataset2: Second dataset, will be used to test
    
    Returns:
        -One DataFrame with all the Scores for each Subject."""
    
    new_y_list = ['PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES',
                  'PUNT_RAZONA_CUANT', 'PUNT_COMP_CIUDADANA', 'PUNT_GLOBAL']
    d1_y_slice = set(dataset1.filter(items=new_y_list))
    d2_y_slice = set(dataset2.filter(items=new_y_list))
    
    y_slice = d1_y_slice & d2_y_slice
    
    LR_List = []
    DT_List = []
    RF_List = []
    idx_List = []
    for i in y_slice:
        LR, LR_Score, DT, DT_Score, RF, RF_Score = Test_with_2sets_2014(dataset1, dataset2, i)
        idx_List.append(i)
        LR_List.append(LR_Score)
        DT_List.append(DT_Score)
        RF_List.append(RF_Score)
    Scores_List = {'Subject': idx_List, 'LR': LR_List, 'DTR': DT_List, 'RFR': RF_List}
    Score_DF = pd.DataFrame(Scores_List)
    return(Score_DF)
def MAD(Y, YP): #MAE - MEAN ABSOLUTE ERROR FOR ME
    """Returns the MEAN ABSOLUTE ERROR between the prediction and the real value"""
    m = np.mean(abs(Y-YP))
    return m

def Prepare_data(data, c):
    """Returns the data ready to be used in a prediction.
    
    -data: full dataset
    -c: column/subject to predict
    
    returns:
        -X2: New Predictive variables powered to 2
        -LR_Data: Old Predictive variables transformed by PCA of 35 components. This will be used for the Linear Regression Object
        -DT_Data: Old Predictive variables transformed by PCA of 33 components. This will be used for the Decision Tree Object
        -Y: Dataframe with the chosen target and the others"""
    data = data.sort_values(by=c)
    
    New_x_list = ['ESTU_GENERO', 'ESTU_ACT_PROX_ANNO', 'COD_INTERDISCIPLINAR', 'COLE_CARACTER', 'ESTU_RESIDE_DEPTO',
                  'FAMI_APORTANTES', 'FAMI_NUM_HERMANOS_EDUSUPERIOR', 'COLE_JORNADA', 'FAMI_OCUPA_MADRE',
                  'ESTU_CARRDESEADA_RAZON', 'FAMI_PERSONAS_HOGAR', 'ESTU_RAZONINSTITUTO', 'FAMI_OCUPA_PADRE',
                  'FAMI_EDUCA_PADRE', 'FAMI_NUM_HERMANOS', 'FAMI_EDUCA_MADRE', 'COLE_VALOR_PENSION', 'ESTU_RESIDE_MCPIO',
                  'ESTU_NACIMIENTO_MES', 'ESTU_IES_COD_DESEADA', 'ESTU_NACIMIENTO_DIA', 'ESTU_NACIMIENTO_ANNO',
                  'ESTU_CARRDESEADA_COD', 'COLE_COD_ICFES', 'FAMI_INGRESO_FMILIAR_MENSUAL']
    
    y_list = ['PUNT_BIOLOGIA', 'PUNT_MATEMATICAS', 'PUNT_FILOSOFIA', 'PUNT_FISICA', 'PUNT_HISTORIA', 'PUNT_QUIMICA', 
              'PUNT_LENGUAJE', 'PUNT_GEOGRAFIA', 'PUNT_INTERDISCIPLINAR', 'PUNT_IDIOMA']
    
    X_list = data.columns.difference(y_list)
    
    New_X_data = data.filter(items = New_x_list)
    Y = data.filter(items = y_list)
    X_data = data.filter(items = X_list)
    
    X2 = New_X_data**2
    
    pca_LR = PCA(n_components=35);
    LR_Data = pca_LR.fit_transform(X_data);
    
    pca_DT = PCA(n_components=33);
    DT_Data = pca_DT.fit_transform(X_data);
    
    return(X2, LR_Data, DT_Data, Y)


def Get_Scores(X2, LR_Data, DT_Data, Y, c):
    """Train the three models (LinearRegression, DecisionTreeRegressor, RandomForestRegressor), and 
    test them. All of this with a cross_val_score object
    
    -X2: New Predictive variables powered to 2
    -LR_Data: Old Predictive variables transformed by PCA of 35 components. This will be used for the Linear Regression Object
    -DT_Data: Old Predictive variables transformed by PCA of 33 components. This will be used for the Decision Tree Object
    -Y: Dataframe with the chosen target and the others
    -c: Chosen column/subject
    
    Returns:
        -Scores(MEAN ABSOLUTE ERROR) of the trained models"""
    
    cv = ShuffleSplit(n = X2.shape[0], n_iter=5, test_size=0.2)
    
    LR = LinearRegression(n_jobs=4)
    LR_scores = cross_val_score(LR, LR_Data, Y[c], scoring='mean_absolute_error', cv = cv, n_jobs=4)
    
    DT = DecisionTreeRegressor(max_depth=6)
    DT_scores = cross_val_score(DT, DT_Data, Y[c], scoring='mean_absolute_error', cv = cv, n_jobs=4)
    
    RF = RandomForestRegressor(max_depth=10, n_jobs=4)
    RF_scores = cross_val_score(RF, X2, Y[c], scoring='mean_absolute_error', cv = cv, n_jobs=4)
    
    return(-LR_scores, -DT_scores, -RF_scores)

def Show_Score(LR_scores, DT_scores, RF_scores, i):
    """Shows the Mean Score and the Standard Deviation of the given scores."""
    print('Scores for the column:', i)
    print('Linear Regression:      ','Mean Score', np.mean(LR_scores), 'STD Scores', np.std(LR_scores))
    print('Decision Tree Regressor:', 'Mean Score', np.mean(DT_scores), 'STD Scores', np.std(DT_scores))
    print('Random Forest:          ', 'Mean Score', np.mean(RF_scores), 'STD Scores', np.std(RF_scores))
    print()
    

    
def Check_C_RFR(lista):
    """Check if the Dataset contains all the required columns to train the Random Forest Regressor, and all the subject to predict.
    -lista: List of the columns of the dataset to be used."""
    
    X_list = ['ESTU_GENERO', 'ESTU_ACT_PROX_ANNO', 'COD_INTERDISCIPLINAR', 'COLE_CARACTER', 'ESTU_RESIDE_DEPTO',
                  'FAMI_APORTANTES', 'FAMI_NUM_HERMANOS_EDUSUPERIOR', 'COLE_JORNADA', 'FAMI_OCUPA_MADRE',
                  'ESTU_CARRDESEADA_RAZON', 'FAMI_PERSONAS_HOGAR', 'ESTU_RAZONINSTITUTO', 'FAMI_OCUPA_PADRE',
                  'FAMI_EDUCA_PADRE', 'FAMI_NUM_HERMANOS', 'FAMI_EDUCA_MADRE', 'COLE_VALOR_PENSION', 'ESTU_RESIDE_MCPIO',
                  'ESTU_NACIMIENTO_MES', 'ESTU_IES_COD_DESEADA', 'ESTU_NACIMIENTO_DIA', 'ESTU_NACIMIENTO_ANNO',
                  'ESTU_CARRDESEADA_COD', 'COLE_COD_ICFES', 'FAMI_INGRESO_FMILIAR_MENSUAL']
    y_list = ['PUNT_BIOLOGIA', 'PUNT_MATEMATICAS', 'PUNT_FILOSOFIA', 'PUNT_FISICA', 'PUNT_HISTORIA', 'PUNT_QUIMICA', 
              'PUNT_LENGUAJE', 'PUNT_GEOGRAFIA', 'PUNT_INTERDISCIPLINAR', 'PUNT_IDIOMA']
    lo = list(set(X_list)-set(lista))
    Tl = list(set(y_list)-set(lista))
    print('Targets Ready') if Tl == 0 else print('There are no ', len(Tl), 'in the DataFrame:', Tl)
    print('Ready to Train') if lo == 0 else print('There are no ', len(lo), 'in the DataFrame:', lo)
    
def Predict_all_Subjects(dataset):
    """Uses the Automation functions to train and test models for each Subject in the given Dataset.
    dataset: Dataset to be used for the models
    
    Returns:
        -One Dataset with all the Scores indexed by subject"""
    
    y_list = ['PUNT_BIOLOGIA', 'PUNT_MATEMATICAS', 'PUNT_FILOSOFIA', 'PUNT_FISICA', 'PUNT_HISTORIA', 'PUNT_QUIMICA', 
              'PUNT_LENGUAJE', 'PUNT_GEOGRAFIA', 'PUNT_INTERDISCIPLINAR', 'PUNT_IDIOMA']
    
    LR_List = []
    DT_List = []
    RF_List = []
    idx_List = []
    for i in y_list:
        X2, LR_Data, DT_Data, Y = Prepare_data(dataset, i)
        LR_scores, DT_scores, RF_scores = Get_Scores(X2, LR_Data, DT_Data, Y, i)
        Show_Score(LR_scores, DT_scores, RF_scores, i)
        idx_List.append(i)
        LR_List.append(np.mean(LR_scores))
        DT_List.append(np.mean(DT_scores))
        RF_List.append(np.mean(RF_scores))
    Scores_List = {'Subject': idx_List, 'LR': LR_List, 'DTR': DT_List, 'RFR': RF_List}
    Scores_DF = pd.DataFrame(Scores_List)
    return(Scores_DF)

#############################################################################################
#############################################################################################

def Test_with_2sets(dataset1, dataset2, c):
    """Train three models with the first Dataset, and Test the models with the second Dataset,
    -Shows the Mean Absolute Error of each model.
    
    -dataset1: First dataset, will be used to train the models
    -dataset2: Second dataset, will be used to test the models
    -c: Column/Subject to predict
    
    Returns:
        -The trained Models
        -The Score of each Model"""
    
    X2, LR_Data, DT_Data, Y = Prepare_data(dataset1, c)
    X2_2, LR_Data_2, DT_Data_2, Y_2 = Prepare_data(dataset2, c)
    
    ##################TRAINING SIDE######################
    LR = LinearRegression(n_jobs=4);
    LR.fit(LR_Data, Y[c]);
    
    DT = DecisionTreeRegressor(max_depth=7);
    DT.fit(DT_Data, Y[c]);
    
    RF = RandomForestRegressor(max_depth=10, n_jobs=4);
    RF.fit(X2, Y[c]);
    ####################################################
    ##################TESTING SIDE######################
    LR_Score = MAD(Y_2[c], LR.predict(LR_Data_2))
    DT_Score = MAD(Y_2[c], DT.predict(DT_Data_2))
    RF_Score = MAD(Y_2[c], RF.predict(X2_2))
    print('Scores for the Columns/Subject:', c)
    print('Score LR:', LR_Score)
    print('Score DT:', DT_Score)
    print('Score RF:', RF_Score)
    print()
    ####################################################
    
    return(LR, LR_Score, DT, DT_Score, RF, RF_Score)

def Test_all_Subjects(dataset1, dataset2):
    """Uses the Test_with_2sets function to do the corresponding predictions for each Subject.
    
    Shows the Prediction of the Subject for each model. 
    
    dataset1: First dataset, will be used to predict
    dataset2: Second dataset, will be used to test
    
    Returns:
        -One DataFrame with all the Scores for each Subject."""
    
    y_list = ['PUNT_BIOLOGIA', 'PUNT_MATEMATICAS', 'PUNT_FILOSOFIA', 'PUNT_FISICA', 'PUNT_HISTORIA', 'PUNT_QUIMICA', 
              'PUNT_LENGUAJE', 'PUNT_GEOGRAFIA', 'PUNT_INTERDISCIPLINAR', 'PUNT_IDIOMA']
    
    LR_List = []
    DT_List = []
    RF_List = []
    idx_List = []
    for i in y_list:
        LR, LR_Score, DT, DT_Score, RF, RF_Score = Test_with_2sets(dataset1, dataset2, i)
        idx_List.append(i)
        LR_List.append(LR_Score)
        DT_List.append(DT_Score)
        RF_List.append(RF_Score)
    Scores_List = {'Subject': idx_List, 'LR': LR_List, 'DTR': DT_List, 'RFR': RF_List}
    Score_DF = pd.DataFrame(Scores_List)
    return(Score_DF)
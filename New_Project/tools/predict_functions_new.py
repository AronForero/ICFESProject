def MAD(Y, YP): #MAE - MEAN ABSOLUTE ERROR FOR ME
    """Returns the mean loss"""
    m = np.mean(abs(Y-YP))
    return m

def plot_predictions_KF(Model, target, X, Y, kf): #PERFECT FUNCTION - DECIDED IF LET IT GET THE SCORES OR NOT
    """Required the Model/Estimator
    The target
    X is the predictive features
    Y is the data that contains all the TARGETS
    and kf is the object of Kfolds
    Returns the Mean Score and the Plots(of each KFold) with the individual score"""
    plot_rows = int(np.ceil(len(kf)/3)) #Number of rows to plot
    fig, axes = plt.subplots(plot_rows, 3, sharex=False, figsize=(14, plot_rows*4))
    axs = [i for i in axes.flatten()]
    sc = []
    j = 0
    for n, i in (kf):
        ax = axs[j]
        Model.fit(X.iloc[n], Y[target].iloc[n])
        Y_pred = Model.predict(X.iloc[i])
        ##################PRUEBA, QUITANDO LOS OUTLIERS EXAGERADOS###############
        #ind = (Y_pred > 0) & (Y_pred < 100)
        #Y_pred2 = Y_pred[ind] 
        score = MAD(Y[target].iloc[i], Y_pred) #HERE THE MAE IS CALCULATED
        ########################################################################
        ax.plot(Y[target].iloc[i], Y_pred, 'go', label='Prediction')
        ax.plot(Y[target].iloc[i], Y[target].iloc[i]) #RECTA DE REGRESION PERFECTA
        ax.set_title('Fold: '+str(j)+'. Score: '+str(score))
        ax.set_xlabel('Real_Value')
        ax.set_ylabel('Prediction')
        ax.legend()
        j += 1
        sc.append(score)
        ax.grid()
    print('Mean Score:', np.mean(sc))
    plt.tight_layout(h_pad=2)
    
def get_exp_prediction(estimator, exp, S_Data, target, folds, KFShuffle = False, shuffle_split = False):
    """Required the estimator
    The exponent
    The sorted data by values of the target to predict
    The column(target) that will be analyzed
    The number of folds that will be created
    KFShuffle if True the Kfolds shuffle the data before split it
    shuffle_split if True the function DON'T use Kfolds, instead use a ShuffleSplit object
    And return the score of the model, and the plot of the prediction"""
    
    y_list=['DECIL_LECTURA_CRITICA', 'PUNT_LECTURA_CRITICA', 'DECIL_MATEMATICAS', 'PUNT_MATEMATICAS', 'DECIL_C_NATURALES',
            'PUNT_C_NATURALES', 'DECIL_SOCIALES_CIUDADANAS', 'PUNT_SOCIALES_CIUDADANAS', 'DECIL_INGLES', 'DESEMP_INGLES',
            'PUNT_INGLES', 'DECIL_RAZONA_CUANT', 'PUNT_RAZONA_CUANT', 'DECIL_COMP_CIUDADANA', 'PUNT_COMP_CIUDADANA', 
            'PUNT_GLOBAL', 'ESTU_PUESTO']
    new_y_list = ['PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES',
                  'PUNT_RAZONA_CUANT', 'PUNT_COMP_CIUDADANA', 'PUNT_GLOBAL']
    X_list = S_Data.columns.difference(y_list)
    New_X = S_Data.filter(items = X_list)
    exped_X = New_X**exp
    Y_train = S_Data.filter(items = y_list)
    
    if shuffle_split == True:
        kf = ShuffleSplit(n = exped_X.shape[0], n_iter=5, test_size=(100/(folds*100)))
    else:
        kf = KFold(exped_X.shape[0], n_folds=folds, shuffle = KFShuffle)
        
    print('The Model with the Data raised to the power of', exp, 'gives:')
    plot_predictions_KF(estimator, target, exped_X, Y_train, kf)
    
def get_Poly_prediction(estimator, deg, D_sorted, target, folds, KFShuffle = False, shuffle_split = False):
    """Required the degree to the PolynomialFeature object and the sorted data by values of the target,
    return the score of the model, statistics and the plots of the prediction"""
    y_list=['DECIL_LECTURA_CRITICA', 'PUNT_LECTURA_CRITICA', 'DECIL_MATEMATICAS', 'PUNT_MATEMATICAS', 'DECIL_C_NATURALES',
            'PUNT_C_NATURALES', 'DECIL_SOCIALES_CIUDADANAS', 'PUNT_SOCIALES_CIUDADANAS', 'DECIL_INGLES', 'DESEMP_INGLES',
            'PUNT_INGLES', 'DECIL_RAZONA_CUANT', 'PUNT_RAZONA_CUANT', 'DECIL_COMP_CIUDADANA', 'PUNT_COMP_CIUDADANA', 
            'PUNT_GLOBAL', 'ESTU_PUESTO']
    new_y_list = ['PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES', 'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES',
                  'PUNT_RAZONA_CUANT', 'PUNT_COMP_CIUDADANA', 'PUNT_GLOBAL']
    X_list = S_Data.columns.difference(y_list)
    New_X = D_sorted.filter(items = X_list)
    Y_train = D_sorted.filter(items = y_list)
    
    Poly = PolynomialFeatures(degree = deg)
    Poly_X = Poly.fit_transform(New_X)
    Poly_X = pd.DataFrame(Poly_X)
    
    if shuffle_split == True:
        kf = ShuffleSplit(n = Poly_X.shape[0], n_iter=5, test_size=(100/(folds*100)))
    else:
        kf = KFold(Poly_X.shape[0], n_folds=folds, shuffle = KFShuffle)
    
    print('The Model with the new "PolyData" with degree', deg, 'gives:')
    plot_predictions_KF(estimator, target, Poly_X, Y_train, kf)
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Mean Absolute Error")
    train_sizes, train_scores, test_scores = learning_curve(estimator,
                                                            X,
                                                            y,
                                                            cv=cv,
                                                            n_jobs=n_jobs,
                                                            scoring = 'mean_absolute_error',
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(-train_scores, axis=1)
    train_scores_std = np.std(-train_scores, axis=1)
    test_scores_mean = np.mean(-test_scores, axis=1)
    test_scores_std = np.std(-test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
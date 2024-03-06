from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def run_logistic(X_train,X_train_, X_test,X_test_,y_train,y_test):
    
    # function to train and test the performance of logistic regression
    logit = LogisticRegression(random_state=0)
    logit.fit(X_train, y_train)
    print('Train set')
    pred = logit.predict_proba(X_train)
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    print('Test set')
    pred = logit.predict_proba(X_test)
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
    
        # function to train and test the performance of logistic regression
    logit_ = LogisticRegression(random_state=0)
    logit_.fit(X_train_, y_train)
    print('Select Train set')
    pred = logit_.predict_proba(X_train_)
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    print('Select Test set')
    pred = logit_.predict_proba(X_test_)
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

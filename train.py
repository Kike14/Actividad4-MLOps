import pandas as pd
import pickle
import optuna

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

def objective(trial):
    data = pd.read_csv("./data/credit_train.csv")
    X = data.drop('Y', axis=1)
    Y = data['Y']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_estimators = trial.suggest_int('n_estimators', 50, 150)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_float('max_features', 0.1, 0.9)
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight='balanced',  # Ajuste para clases desbalanceadas
        random_state=1234
    )

    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=1234)
    f1_scores = cross_val_score(model, X, Y, cv=kfold, scoring='f1')

    return f1_scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("Mejores hiperparámetros encontrados: ", study.best_params)

def main():
    data = pd.read_csv("./data/credit_train.csv")
    X = data.drop('Y', axis=1)
    Y = data['Y']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1234)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_params = study.best_params

    model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        bootstrap=best_params['bootstrap'],
        class_weight='balanced',  # Ajuste para clases desbalanceadas
        random_state=1234
    ).fit(X_train, Y_train)

    Y_hat_train = model.predict(X_train)
    Y_hat_test = model.predict(X_test)

    f1_train = f1_score(Y_train, Y_hat_train)
    f1_test = f1_score(Y_test, Y_hat_test)

    accuracy_train = accuracy_score(Y_train, Y_hat_train)
    accuracy_test = accuracy_score(Y_test, Y_hat_test)

    print("F1 score train: ", f1_train)
    print("F1 score test: ", f1_test)
    print("Accuracy train: ", accuracy_train)
    print("Accuracy test: ", accuracy_test)

    with open("./models/random_forest.pkl", "wb") as file:
        pickle.dump(model, file)

    with open("./models/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

if __name__ == '__main__':
    main()

from sklearn.metrics import mean_squared_error

from src.pipelines.training.helpers import compute_classification_metrics, compute_regression_metrics


def training_pipeline(config, model, train_dataset, test_dataset):
    
    X_train, Y_train, X_train_ids = [], [], []
    X_test, Y_test, X_test_ids = [], [], []
    
    for i in range(train_dataset.len()):
        id, input, output = train_dataset.get(i)
        X_train.append(input)
        Y_train.append(output)
        X_train_ids.append(id)
    
    for i in range(test_dataset.len()):
        id, input, output = test_dataset.get(i)
        X_test.append(input)
        Y_test.append(output)
        X_test_ids.append(id)


    model.fit(X_train, Y_train)

    stats = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1_micro": [],
        "test_accuracy": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1_micro": [],
        "train_r2": [],
        "test_r2": [],
        "learning_rate": [],
        "last_epoch_train_data": None,
        "last_epoch_test_data": None,
        "equation": None
    }
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    
    if config['task'].get('num_classes') > 1:
        Y_train_pred_class = (Y_train_pred >= 0.5).astype(int)
        Y_test_pred_class = (Y_test_pred >= 0.5).astype(int)

    train_loss = mean_squared_error(Y_train, model.predict(X_train))
    test_loss = mean_squared_error(Y_test, model.predict(X_test))
    
    if config['task'].get('num_classes') > 1:
        train_acc, train_prec, train_rec, train_f1_micro, train_r2 = compute_classification_metrics(Y_train, Y_train_pred_class)
        test_acc, test_prec, test_rec, test_f1_micro, test_r2 = compute_classification_metrics(Y_test, Y_test_pred_class)
    else:
        train_acc, train_prec, train_rec, train_f1_micro, train_r2 = compute_regression_metrics(Y_train, Y_train_pred)
        test_acc, test_prec, test_rec, test_f1_micro, test_r2 = compute_regression_metrics(Y_test, Y_test_pred)

    stats["epoch"].append(0)
    stats["train_loss"].append(train_loss)
    stats["test_loss"].append(test_loss)
    stats["train_accuracy"].append(train_acc)
    stats["train_precision"].append(train_prec)
    stats["train_recall"].append(train_rec)
    stats["train_f1_micro"].append(train_f1_micro)
    stats["test_accuracy"].append(test_acc)
    stats["test_precision"].append(test_prec)
    stats["test_recall"].append(test_rec)
    stats["test_f1_micro"].append(test_f1_micro)
    stats["train_r2"].append(train_r2)
    stats["test_r2"].append(test_r2)
    stats["learning_rate"].append(None)
    if config['model']['type'] == 'symbolic_regression':
        stats["equation"] = model.latex()

    stats["last_epoch_train_data"] = list(zip(X_train_ids, Y_train, Y_train_pred_class ))
    stats["last_epoch_test_data"] = list(zip(X_test_ids, Y_test, Y_test_pred_class ))  

    return model, stats
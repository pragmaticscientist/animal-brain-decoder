import torch
from src.pipelines.training.helpers import compute_classification_metrics, compute_regression_metrics, is_classification

def training_loop(config, model, train_loader, test_loader, optimizer, criterion, scheduler, device):
    epochs = config['training'].get('epochs', 100)

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

    for epoch in range(epochs):
        train_loss, train_acc, train_prec, train_rec, train_f1, train_r2, last_epoch_train_data = train_one_epoch(config, model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, test_prec, test_rec, test_f1, test_r2, last_epoch_test_data = evaluate(config, model, test_loader, criterion, device)

        if epoch == epochs - 1:
            stats["last_epoch_train_data"] = last_epoch_train_data
            stats["last_epoch_test_data"] = last_epoch_test_data

        stats["epoch"].append(epoch)
        stats["train_loss"].append(train_loss)
        stats["test_loss"].append(test_loss)
        stats["train_accuracy"].append(train_acc)
        stats["train_precision"].append(train_prec)
        stats["train_recall"].append(train_rec)
        stats["train_f1_micro"].append(train_f1)
        stats["test_accuracy"].append(test_acc)
        stats["test_precision"].append(test_prec)
        stats["test_recall"].append(test_rec)
        stats["test_f1_micro"].append(test_f1)
        stats["train_r2"].append(train_r2)
        stats["test_r2"].append(test_r2)
        stats["learning_rate"].append(optimizer.param_groups[0]['lr'])

        if is_classification(config):        
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train R^2: {train_r2:.4f}, Test R^2: {test_r2:.4f}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(test_loss)
        else:
            scheduler.step()
    return stats

def train_one_epoch(config, model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_ids = []

    for id, input, output in train_loader:
        input = input.to(device)
        true_output = output.to(device)
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, true_output)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()*input.size(0)  # Multiply by batch size to get total loss for the epoch 
        if is_classification(config):
            preds = outputs.argmax(dim=1).cpu().numpy()
        else:
            preds = outputs
        labels = output.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_ids.extend(id)

    avg_loss = total_loss / len(train_loader.dataset)
    
    if is_classification(config):
        acc, prec, rec, f1_micro, r2 = compute_classification_metrics(all_labels, all_preds)
    else:
        acc, prec, rec, f1_micro, r2 = compute_regression_metrics(all_labels, all_preds)

    return avg_loss, acc, prec, rec, f1_micro, r2, list(zip(all_ids, all_labels, all_preds)) 

def evaluate(config, model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for id, input, output in test_loader:
            input = input.to(device)
            true_output = output.to(device)
            outputs = model(input)
            loss = criterion(outputs, true_output)

            total_loss += loss.item()*input.size(0)  # Multiply by batch size to get total loss for the epoch 
            if is_classification(config):
                preds = outputs.argmax(dim=1).cpu().numpy()
            else:
                preds = outputs
            labels = output.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_ids.extend(id)

    avg_loss = total_loss / len(test_loader.dataset)
    if is_classification(config):
        acc, prec, rec, f1_micro, r2 = compute_classification_metrics(all_labels, all_preds)
    else:
        acc, prec, rec, f1_micro, r2 = compute_regression_metrics(all_labels, all_preds)

    return avg_loss, acc, prec, rec, f1_micro, r2, list(zip(all_ids, all_labels, all_preds))


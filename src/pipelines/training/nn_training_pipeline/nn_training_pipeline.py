import torch
from src.models import model_factory
from src.pipelines.training.nn_training_pipeline.training_loop import training_loop

def training_pipeline(config, model, train_dataset, test_dataset):
    # Training setup
    
    # Device configuration
    device, num_gpus = get_device(config['training'])

    if num_gpus > 1 and config['model']['type'] != 'mlp':
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Optimizer and loss function
    optimizer = get_optimizer(config['training'].get('optimizer'), model)
    criterion = get_loss_function(config['training'])
    scheduler = get_scheduler(config['training'].get('scheduler'), optimizer)

     # Data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    stats = training_loop(config['training'], model, train_loader, test_loader, optimizer, criterion, scheduler, device)

    # save stats
    
    return model, stats

def get_device(config):
    gpu = config.get('gpu', False)
    if torch.cuda.is_available() and gpu:
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPU(s)")
    else:
        device = torch.device("cpu")
        num_gpus = 0
        print("Using CPU")

    return device, num_gpus

def get_optimizer(config, model):
    learning_rate = config.get('learning_rate', 0.001)
    optimizer_type = config.get('optimizer', 'adam').lower()

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer

def get_scheduler(config, optimizer):
    scheduler_type = config.get('type', 'plateau').lower()
    step_size = config.get('scheduler_step_size', 10)
    gamma = config.get('scheduler_gamma', 0.1)

    if scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=step_size)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler

def get_loss_function(config):
    loss_type = config.get('loss_function', 'cross_entropy').lower()

    if loss_type == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_type == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function type: {loss_type}")

    return criterion
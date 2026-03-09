from src.pipelines.data_preparation_pipeline import data_preparation_pipeline 
from src.pipelines.stat_handling_pipeline import stat_handling_pipeline
from src.pipelines.training.training_pipeline_dispatcher import training_pipeline_dispatcher
#from src.pipelines.model_handling_pipeline  import model_handling_pipeline
from src.models import model_factory

def run_pipeline(config):
    n_runs = config.get("runs", 1)
    for run in range(n_runs):
        master_pipeline(config, run)

def master_pipeline(config, run):
    print(f"Running master pipeline with config (Run {run + 1}):")
    train_dataset, test_dataset = data_preparation_pipeline(config['data'], config['split'], config['task'])
    model = model_factory.get_model(config['model'])
    model, stats = training_pipeline_dispatcher(config, model, train_dataset, test_dataset)
    stat_handling_pipeline(config["statistics"], run, stats)
    """
    model_handling_pipeline(config, model)
    """

from src.pipelines.data_preparation_pipeline import data_preparation_pipeline 
#from src.pipelines.stat_handling_pipeline import stat_handling_pipeline
from src.pipelines.training.training_pipeline_dispatcher import training_pipeline_dispatcher
#from src.pipelines.model_handling_pipeline  import model_handling_pipeline
from src.models import model_factory

def run_pipeline(config):
    master_pipeline(config)

def master_pipeline(config):
    print("Running master pipeline with config:")
    train_dataset, test_dataset = data_preparation_pipeline(config['data'])
    model = model_factory.get_model(config['model'])
    model, stats = training_pipeline_dispatcher(config, model, train_dataset, test_dataset)
    """
    stat_handling_pipeline(config, stats)
    model_handling_pipeline(config, model)
    """

import hydra
from omegaconf import DictConfig, OmegaConf
from src.pipelines import master_pipeline

# The @hydra.main decorator takes over the role of argparse.
# - config_path points to the directory containing your configs (e.g., 'conf/')
# - config_name is the default yaml file to load (without the .yaml extension)
@hydra.main(version_base=None, config_path="../config")
def main(cfg: DictConfig):
    
    # Optional: If your run_pipeline function requires a standard Python dictionary
    # rather than a Hydra DictConfig object, you can convert it like this.
    # The `resolve=True` argument ensures all interpolations (like ${task.name}) are calculated.
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    print("=== FULL COMPOSED CONFIG ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("============================")
    
    # Run your pipeline
    master_pipeline.run_pipeline(config_dict)

if __name__ == "__main__":
    # You just call main() directly. Hydra will automatically inject 'cfg'.
    main()

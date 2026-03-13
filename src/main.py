import hydra
from omegaconf import DictConfig, OmegaConf
from src.pipelines import master_pipeline

# - config_path points to the directory containing your configs (e.g., 'conf/')
# - config_name is the default yaml file to load (without the .yaml extension)
@hydra.main(version_base=None, config_path="../config")
def main(cfg: DictConfig):
    
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    print("=== FULL COMPOSED CONFIG ===")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("============================")
    
    # Run your pipeline
    master_pipeline.run_pipeline(config_dict)

if __name__ == "__main__":
    main()

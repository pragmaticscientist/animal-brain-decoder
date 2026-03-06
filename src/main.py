import argparse
import yaml
from src.pipelines import master_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipelines based on a configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file in YAML format",
    )

    args = parser.parse_args()
    config_file = args.config

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    master_pipeline.run_pipeline(config)
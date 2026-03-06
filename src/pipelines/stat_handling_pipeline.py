import pandas as pd    

def stat_handling_pipeline(config, stats):
    """
    Save statistics to CSV files based on config specifications.
    
    Args:
        config: Configuration dictionary with keys:
            - 'stats_output_file': Path to save metrics and general stats
            - 'predictions_output_file': Path to save last epoch train/test data
        stats: Statistics dictionary containing metrics and prediction data
    """
    # Separate stats into metrics and prediction data
    metrics_stats = {k: v for k, v in stats.items() 
                     if k not in ["last_epoch_train_data", "last_epoch_test_data"]}
    
    # Save metrics to CSV
    stats_file = config.get('stats_output_file')
    if stats_file:
        metrics_df = pd.DataFrame(metrics_stats)
        metrics_df.to_csv(stats_file, index=False)
        print(f"Saved metrics to {stats_file}")
    
    # Save prediction data to CSV
    predictions_file = config.get('predictions_output_file')
    if predictions_file:
        train_data = stats["last_epoch_train_data"]
        test_data = stats["last_epoch_test_data"]
        
        # Combine train and test data with a split indicator
        all_predictions = []
        if train_data:
            for item in train_data:
                all_predictions.append({"split": "train", "id": item[0], "ground_truth": item[1], "prediction": item[2]})
        if test_data:
            for item in test_data:
                all_predictions.append({"split": "test", "id": item[0], "ground_truth": item[1], "prediction": item[2]})
        
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Saved prediction data to {predictions_file}")  
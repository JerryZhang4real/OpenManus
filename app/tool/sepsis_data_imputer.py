
# help me to impute missing values in a CSV file located at "/Users/apple/GitHubRepos/OpenManus/data/mimic/raw/full_dataset.csv" using the modle located at "/Users/apple/GitHubRepos/OpenManus/experiments/experiment_48851.ckpt"

from pathlib import Path
from typing import Optional
import pandas as pd
import torch

# Import BaseTool and ToolResult from the OpenManus tool base module
from app.tool.base import BaseTool, ToolResult

# Import model loading and data parsing utilities
from run_samples import Inspector
from lib.parse_datasets import parse_datasets
import lib.utils as utils
from torch.utils.data import DataLoader

class SepsisDataImputer(BaseTool):
    name: str = "spesis_data_imputer"
    description: str = (
        "Load a time-series forecasting model from a checkpoint and perform inference on a CSV file. "
        "Uses the given model checkpoint to forecast values for the data in the CSV. "
        "Optionally, a specific record ID can be provided to filter the dataset before running inference. "
        "The tool returns the model's predictions (forecasts) for the input data."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "ckpt_path": {
                "type": "string",
                "description": "The full path to the model checkpoint file to load."
            },
            "csv_file_path": {
                "type": "string",
                "description": "The full path to the input CSV data file for forecasting."
            },
            "record_id": {
                "type": "string",
                "description": "Optional record ID to filter the CSV data (e.g., a specific patient ID)."
            }
        },
        "required": ["ckpt_path", "csv_file_path"]
    }

    async def execute(self, **kwargs) -> ToolResult:
        ckpt_path: Optional[str] = kwargs.get("ckpt_path")
        csv_file_path: Optional[str] = kwargs.get("csv_file_path")
        record_id = kwargs.get("record_id")  # can be int, float, or str

        # Validate required parameters
        if not ckpt_path:
            return ToolResult(error="ckpt_path is required.")
        if not csv_file_path:
            return ToolResult(error="csv_file_path is required.")

        # Validate file existence
        model_path = Path(ckpt_path)
        data_path = Path(csv_file_path)
        if not model_path.is_file():
            return ToolResult(error=f"Model checkpoint not found at {ckpt_path}. Please check the path.")
        if not data_path.is_file():
            return ToolResult(error=f"CSV file not found at {csv_file_path}. Please check the path.")

        try:
            # 1. Load the model checkpoint (use CPU by default)
            print(f"Loading model from: {ckpt_path}")
            device = torch.device("cpu")  # default to CPU for inference
            model, args = Inspector.load_ckpt(ckpt_path, device)
            print("Model loaded successfully.")

            # 2. Parse the dataset to prepare data loaders and metadata
            args.batch_size = 1
            args.n = 100  # limit number of samples for parsing (adjust as needed)
            print("Parsing dataset for model input...")
            data_obj = parse_datasets(args, patch_ts=True)
            print("Dataset parsed into data loaders.")

            # 3. Load the CSV data using pandas
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            print(f"CSV file loaded: {len(df)} rows read.")
            # If a specific ID is provided, filter the DataFrame
            if record_id is not None:
                try:
                    # Attempt numeric comparison (ID might be float/int in data)
                    df_filtered = df[df['HADM_ID'] == float(record_id)]
                except Exception:
                    # Fallback to direct match (e.g., if ID is string-type)
                    df_filtered = df[df['HADM_ID'] == record_id]
                if df_filtered.empty:
                    return ToolResult(error=f"No data found for ID {record_id} in the CSV file.")
                df = df_filtered
                print(f"Data filtered for record ID {record_id}: {len(df)} rows remaining.")

            # 4. Run model inference on the prepared data
            print("Starting model inference...")
            predictions = []
            try:
                # By default, use the training data loader for inference (as in test.py)
                dataloader = data_obj["train_dataloader"]
                n_batches = data_obj.get("n_train_batches", 0)
                # If a specific record was requested, attempt to isolate that record in the dataset
                if record_id is not None:
                    # Try to find the record in the combined dataset (train/val/test)
                    try:
                        # Reconstruct dataset lists from data_obj if possible
                        train_loader = data_obj["train_dataloader"]  # infinite generator
                        val_loader = data_obj.get("val_dataloader")
                        test_loader = data_obj.get("test_dataloader")
                        # If the infinite generators have underlying dataset lists accessible (not guaranteed)
                        dataset_records = []
                        if hasattr(train_loader, "dataset"):
                            dataset_records += list(train_loader.dataset)
                        if val_loader and hasattr(val_loader, "dataset"):
                            dataset_records += list(val_loader.dataset)
                        if test_loader and hasattr(test_loader, "dataset"):
                            dataset_records += list(test_loader.dataset)
                        # Filter for the target record
                        target_records = [rec for rec in dataset_records 
                                          if rec and (rec[0] == record_id or rec[0] == float(record_id))]
                        if target_records:
                            # Create a temporary DataLoader for just this record (batch_size=1)
                            collate_fn = None
                            if hasattr(train_loader, "collate_fn"):
                                collate_fn = train_loader.collate_fn  # reuse existing collate function
                            single_loader = DataLoader(target_records, batch_size=1, shuffle=False, collate_fn=collate_fn)
                            dataloader = utils.inf_generator(single_loader)  # wrap in infinite generator like others
                            n_batches = 1
                            print(f"Isolated record {record_id} for inference.")
                        else:
                            print(f"Record ID {record_id} not found in parsed dataset; using full train data for inference.")
                    except Exception as isolate_err:
                        print(f"Warning: could not isolate record {record_id} - {isolate_err}")
                # Iterate through the batches and collect model predictions&#8203;
                for _ in range(n_batches):
                    batch = utils.get_next_batch(dataloader)
                    if batch is None:
                        break
                    # Call the model's forecasting method with batch data
                    pred_y = model.forecasting(batch["tp_to_predict"],
                                               batch["observed_data"],
                                               batch["observed_tp"],
                                               batch["observed_mask"])
                    # Convert prediction to a Python data type for output (e.g., list or string)
                    try:
                        pred_array = pred_y.detach().cpu().numpy()
                        predictions.append(pred_array.tolist())
                    except Exception:
                        # If conversion fails, just append the raw prediction object
                        predictions.append(str(pred_y))
            except Exception as inf_err:
                # Catch any unexpected errors during inference
                return ToolResult(error=f"Error during inference: {str(inf_err)}")

            # 5. Format the output results
            output_lines = []
            if record_id is not None:
                # Include the filtered data (as string) for context if an ID filter was applied
                output_lines.append(f"Filtered data for ID {record_id}:\n{df.to_string(index=False)}")
            # Include the model predictions
            if predictions:
                output_lines.append("Model predictions:")
                for pred in predictions:
                    output_lines.append(str(pred))
            else:
                output_lines.append("No predictions were generated.")
            output_text = "\n\n".join(output_lines)
            return ToolResult(output=output_text)
        except Exception as e:
            # Handle any other errors and provide a message
            return ToolResult(error=f"Failed to run ForecastFromFile: {str(e)}")

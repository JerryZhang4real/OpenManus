"""
predict_sepsis_tool.py

A BaseTool implementation that loads a trained mortality prediction model and applies it to an input CSV file.

test prompt: Help me predict the mortality of samples located in "/Users/apple/GitHubRepos/OpenManus/AI_agent_test_sepsis_features.csv"
"""
import os
from pathlib import Path
import pandas as pd
import joblib

# Import BaseTool and ToolResult from the project's base tool module.
from app.tool.base import BaseTool, ToolResult

class MortalityPrediction(BaseTool):
    name: str = "mortality_prediction"
    description: str = (
        "Read a CSV file of sepsis features, load a pre-trained Random Forest mortality model, "
        "perform inference, and save predictions back to disk."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Full path to the CSV file containing sepsis features.",
            },
        },
        "required": ["file_path"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        file_path = kwargs.get("file_path")
        if not file_path:
            return ToolResult(error="'file_path' is required.")

        input_path = Path(file_path)
        if not input_path.is_file():
            return ToolResult(error=f"File not found at {file_path}.")

        # Default model location
        model_path = Path("/Users/apple/GitHubRepos/OpenManus/model/sepsis_rf_model.pkl")
        if not model_path.is_file():
            return ToolResult(error=f"Model file not found at {model_path}.")

        try:
            # 1. load data
            df = pd.read_csv(input_path)
            
            # 2. load the full pipeline (imputer, selector, scaler, RF classifier)
            pipeline = joblib.load(model_path)
            # ensure pipeline has predict and predict_proba
            if not hasattr(pipeline, "predict"):
                return ToolResult(error="Loaded object has no `predict` method; check your pickle.")

            # 3. run inference
            #    - mortality_proba: probability of class 1
            proba = pipeline.predict_proba(df)[:, 1]
            #    - mortality_pred: hard label 0/1
            preds = pipeline.predict(df)
            
            # 4. append to DataFrame
            df["mortality_proba"] = proba
            df["mortality_pred"] = preds
            
            # 5. save to new file
            out_path = input_path.with_name(input_path.stem + "_predictions.csv")
            df.to_csv(out_path, index=False) 

            return ToolResult(output=f"Predictions saved to {out_path}")
        except Exception as e:
            return ToolResult(error=f"Error during prediction: {e}")

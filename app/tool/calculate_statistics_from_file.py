import os
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

# Import BaseTool and ToolResult from the project's base tool module.
from app.tool.base import BaseTool, ToolResult

class CalculateStatisticsFromFile(BaseTool):
    name: str = "calculate_statistics_from_file"
    description: str = (
        "Read a CSV file from the local computer and calculate statistical measures for numeric data. "
        "Computes count, mean, median, standard deviation, min, max, and quartiles for each numeric column."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The full path of the CSV file to read from the local computer.",
            },
        },
        "required": ["file_path"],
    }
    
    async def execute(self, **kwargs) -> ToolResult:
        file_path: Optional[str] = kwargs.get("file_path")
        if not file_path:
            return ToolResult(error="file_path is required.")
        
        # Create a Path object for the file.
        path = Path(file_path)
        if not path.is_file():
            return ToolResult(error=f"File not found at {file_path}. Please check the path and try again.")
        
        try:
            # Read the CSV file into a pandas DataFrame.
            df = pd.read_csv(file_path)
            
            # Calculate basic statistics using describe, which includes count, mean, std, min, quartiles, and max.
            stats_df = df.describe(include="number")
            
            # Calculate the median explicitly for each numeric column.
            median_series = df.median(numeric_only=True)
            stats_df.loc['median'] = median_series
            
            # Optionally, sort the index for a clean output order.
            stats_df = stats_df.sort_index()
            
            # Convert the DataFrame with statistics to a dictionary.
            stats = stats_df.to_string()
            
            return ToolResult(output=stats)
        except Exception as e:
            return ToolResult(error=f"Error processing the file: {str(e)}")

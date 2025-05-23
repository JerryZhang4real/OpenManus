from typing import Any

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.file_saver import FileSaver
from app.tool.web_search import WebSearch
from app.tool.python_execute import PythonExecute
from app.config import config

from app.tool.download_file import DownloadFile
from app.tool.calculate_statistics_from_file import CalculateStatisticsFromFile
from app.tool.sepsis_data_imputer import SepsisDataImputer
from app.tool.mortality_prediction import MortalityPrediction

class Manus(ToolCallAgent):
    """
    A versatile general-purpose agent that uses planning to solve various tasks.

    This agent extends PlanningAgent with a comprehensive set of tools and capabilities,
    including Python execution, web browsing, file operations, and information retrieval
    to handle a wide range of user requests.
    """

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 2000
    max_steps: int = 20

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), WebSearch(), BrowserUseTool(), FileSaver(), Terminate(), DownloadFile(), CalculateStatisticsFromFile(), SepsisDataImputer(), MortalityPrediction()
        )
    )

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        await self.available_tools.get_tool(BrowserUseTool().name).cleanup()
        await super()._handle_special_tool(name, result, **kwargs)

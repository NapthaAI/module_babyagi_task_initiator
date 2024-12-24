from pydantic import BaseModel, Field
from typing import List

class TaskInitiatorPromptSchema(BaseModel):
    """Schema for task initiator input data"""
    objective: str
    context: str = Field(default="", description="Optional context for task generation")

class InputSchema(BaseModel):
    """Input schema matching the task executor's structure"""
    tool_name: str = Field(default="generate_tasks", description="Name of the method to call")
    tool_input_data: TaskInitiatorPromptSchema
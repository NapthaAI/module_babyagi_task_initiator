from dotenv import load_dotenv
from babyagi_task_initiator.schemas import (
    InputSchema,
    TaskInitiatorPromptSchema,
    Task,
    TaskList
)
from naptha_sdk.schemas import AgentDeployment, AgentRunInput
from naptha_sdk.utils import get_logger
import json
import os
from litellm import completion
load_dotenv()
logger = get_logger(__name__)

class TaskInitiatorAgent:
    def __init__(self, agent_deployment: AgentDeployment):
        self.agent_deployment = agent_deployment

        self.user_message_template = """
                Here’s an improved version of your prompt:

                You are given the following task: {{task}}. The goal is to accomplish the following objective: {{objective}}.

                Instructions:

                    Break down the task into specific, actionable steps, and populate the name and description fields accordingly.
                    Use the done field to indicate whether the task has been completed (True if completed, False otherwise).
                    Provide a clear and concise summary of the outcome in the result field.

                Ensure that the result field reflects the actual outcome of the task after completion, and the done field accurately represents the task's status.
                """

    def generate_tasks(self, inputs: InputSchema) -> str:
        user_prompt = self.user_message_template.replace(
            "{{objective}}",
            inputs.tool_input_data.objective
        )

        # Prepare context if available
        context = inputs.tool_input_data.context
        if context:
            user_prompt += f"\nContext: {context}"

        # Prepare messages
        messages = [
            {"role": "system", "content": json.dumps(self.agent_deployment.agent_config.system_prompt)},
            {"role": "user", "content": user_prompt}
        ]

        # Prepare LLM configuration
        llm_config = self.agent_deployment.agent_config.llm_config
        api_key = None if llm_config.client == "ollama" else ("EMPTY" if llm_config.client == "vllm" else os.getenv("OPENAI_API_KEY"))

        task = Task(name="Write a blog post", description="Write a blog post about the weather in London.", done=False, result="")

        # Make LLM call
        response = completion(
            model=llm_config.model,
            messages=messages,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
            api_base=llm_config.api_base,
            api_key=api_key,
            response_format=TaskList
        )
            
        # Parse the response content into tasks
        response_content = response.choices[0].message.content
        try:
            print(response_content)
            response_content = json.loads(response_content)
            # Extract the list of tasks
            tasks_data = response_content.get("list", [])
            
            # Ensure tasks_data is a list of dictionaries
            if not isinstance(tasks_data, list) or not all(isinstance(task, dict) for task in tasks_data):
                raise ValueError("Invalid task structure: Expected a list of dictionaries in 'list' key.")
            
            # Create Task objects
            tasks = TaskList(list=[
                Task(**task) for task in tasks_data
            ])
            
            # Log and return the tasks as JSON
            logger.info(f"Generated Tasks: {tasks.model_dump_json()}")
            return tasks.model_dump_json()
        except json.JSONDecodeError:
            logger.error(f"Failed to parse response as JSON: {response_content}")
            # Return an empty task list if parsing fails
            return TaskList().model_dump_json()

def run(agent_run: AgentRunInput, *args, **kwargs):
    logger.info(f"Running with inputs {agent_run.inputs.tool_input_data}")
    task_initiator_agent = TaskInitiatorAgent(agent_run.agent_deployment)
    method = getattr(task_initiator_agent, agent_run.inputs.tool_name, None)
    return method(agent_run.inputs)

if __name__ == "__main__":
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import load_agent_deployments

    naptha = Naptha()
    
    # Load agent deployments
    agent_deployments = load_agent_deployments(
        "babyagi_task_initiator/configs/agent_deployments.json",
        load_persona_data=False,
        load_persona_schema=False
    )

    # Prepare input parameters
    input_params = InputSchema(
        tool_name="generate_tasks",
        tool_input_data=TaskInitiatorPromptSchema(
            objective="Write a blog post about the weather in London.",
            context="Focus on historical weather patterns between 1900 and 2000"
        )
    )

    # Create agent run input
    agent_run = AgentRunInput(
        inputs=input_params,
        agent_deployment=agent_deployments[0],
        consumer_id=naptha.user.id,
    )

    # Run the agent
    response = run(agent_run)
    logger.info(f"Final Response: {response}")
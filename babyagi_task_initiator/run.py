from dotenv import load_dotenv
from babyagi_task_initiator.schemas import (
    InputSchema,
    TaskInitiatorPromptSchema
)
from naptha_sdk.schemas import AgentDeployment, AgentRunInput
from naptha_sdk.utils import get_logger
import json
from litellm import completion
import asyncio

load_dotenv()
logger = get_logger(__name__)

class TaskInitiatorAgent:
    def __init__(self, agent_deployment: AgentDeployment):
        self.agent_deployment = agent_deployment

        self.user_message_template = """
                You are given the following task: {{task}}. The goal is to accomplish the following objective: {{objective}}.

                Instructions:

                    Break down the task into specific, actionable steps, and populate the name and description fields accordingly.
                    Use the done field to indicate whether the task has been completed (True if completed, False otherwise).
                    Provide a clear and concise summary of the outcome in the result field.

                Ensure that the result field reflects the actual outcome of the task after completion, and the done field accurately represents the task's status.
                """

    async def generate_tasks(self, inputs: InputSchema) -> str:
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
            {"role": "system", "content": json.dumps(self.agent_deployment.config.system_prompt)},
            {"role": "user", "content": user_prompt}
        ]

        # Prepare LLM configuration
        llm_config = self.agent_deployment.config.llm_config

        input_ = {
            "messages": messages,
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens
        }

        response = await naptha.node.run_inference(
            input_
        )

        try:
            response_content = response.choices[0].message.content
            return response_content
        
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse response: {response}. Error: {e}")
            return
        

async def run(agent_run: AgentRunInput, *args, **kwargs):
    logger.info(f"Running with inputs {agent_run.inputs.tool_input_data}")
    task_initiator_agent = TaskInitiatorAgent(agent_run.deployment)
    method = getattr(task_initiator_agent, agent_run.inputs.tool_name, None)
    return await method(agent_run.inputs)

if __name__ == "__main__":
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import setup_module_deployment
    import os

    naptha = Naptha()
    
    # Load agent deployments
    deployment = asyncio.run(setup_module_deployment("agent", "babyagi_task_initiator/configs/deployment.json", node_url = os.getenv("NODE_URL")))

    deployment = AgentDeployment(**deployment.model_dump())

    print("BabyAGI Task Initiator Deployment:", deployment)

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
        deployment=deployment,
        consumer_id=naptha.user.id,
    )

    # Run the agent
    response = asyncio.run(run(agent_run))
    logger.info(f"Final Response: {response}")
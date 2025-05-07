import inspect
import json
from typing import List, Dict, Any, Callable

from datasets import Dataset

from verifiers import RewardFunc
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.prompts import MEMORY_AGENT_PROMPT
from verifiers.rubrics.memory_rubric import MemoryRubric

from obsidian_agent.engine import execute_sandboxed_code
from obsidian_agent.utils import create_memory_if_not_exists
from obsidian_agent.settings import MEMORY_PATH

# Define constants
STOP_TOKENS = ["</python>", "</answer>"]
MASK_ENV_RESPONSE = True
MAX_STEPS = 10
TOOLS_MODULE = "obsidian_agent.tools"

class ObsidianAgentEnv(MultiTurnEnv):
    """
    Environment for the Obsidian Agent.
    """
    def __init__(
            self,
            dataset: Dataset | None = None,
            eval_dataset: Dataset | None = None,
            tools: List[Callable] = [],
            system_prompt: str = MEMORY_AGENT_PROMPT,
            few_shot: List[Dict[str, str]] = [],
            sampling_args={
                "stop": STOP_TOKENS,
                "include_stop_str_in_output": True
            },
            mask_env_response: bool = MASK_ENV_RESPONSE,
            max_steps: int = MAX_STEPS,
            **kwargs: Any
        ):
        
        # Initialize the environment
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            max_steps=max_steps,
            sampling_args=sampling_args,
            **kwargs
        )
        self.dataset_name = dataset
        self.max_steps = max_steps
        self.llm_parser = XMLParser(fields=["thoughts", ("python", "answer")])
        self.env_parser = XMLParser(fields=["result"])
        self.rubric = MemoryRubric()

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        """
        Returns a list of reward functions for the environment.
        """
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        """
        Returns a list of reward weights for the environment.
        """
        return self.rubric.get_reward_weights()

    def execute_python_code(self, code: str) -> str:
        """
        Execute the given python code in a sandboxed environment.

        Args:
            code: The python code to execute.

        Returns:
            The output of the python code.
        """
        create_memory_if_not_exists()
        return execute_sandboxed_code(
            code=code,
            allowed_path=MEMORY_PATH,
            import_module=TOOLS_MODULE
        )
    
    def is_completed(self, messages: List[Dict[str, str]], **kwargs: Any) -> bool:
        pass
    
    def env_response(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, str]:
        pass
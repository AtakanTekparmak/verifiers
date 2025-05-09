import os
import json
from pathlib import Path
from datasets import Dataset
import argparse

from verifiers.envs.memory_agent.obsidian_agent_env import ObsidianAgentEnv
from verifiers.trainers import GRPOEnvTrainer
from verifiers.utils.utils import (
    make_dataset_from_contexts,
    add_stop_tokens,
    compute_reward,
    compute_all_rewards,
)
from trl import GRPOConfig

def load_convo_data(convo_file_path):
    """
    Load conversation data from the JSON file.
    
    Args:
        convo_file_path: Path to the JSON file containing conversations
        
    Returns:
        A list of dictionaries containing the conversation data
    """
    with open(convo_file_path, "r") as f:
        data = json.load(f)
    
    # Format data for training
    formatted_data = []
    
    for convo in data.get("convos", []):
        persona_name = convo.get("user_persona_name_surname", "")
        chats = convo.get("user_chats", [])
        
        for chat in chats:
            turn_number = chat.get("turn_number", 0)
            base_fact = chat.get("base_fact", "")
            user_message = chat.get("user_message", "")
            facts_to_check = chat.get("facts_to_check_so_far", [])
            
            # Create training example
            formatted_data.append({
                "persona_name": persona_name,
                "turn_number": turn_number,
                "base_fact": base_fact,
                "user_message": user_message,
                "facts_to_check_so_far": facts_to_check,
                "context": user_message,  # This will be the main input to the model
            })
    
    return formatted_data

def main():
    parser = argparse.ArgumentParser(description="Train Memory Agent")
    parser.add_argument("--data_path", type=str, default="data/convos.json", 
                        help="Path to conversation data JSON file")
    parser.add_argument("--output_dir", type=str, default="results/memory_agent", 
                        help="Directory to save model and results")
    parser.add_argument("--lr", type=float, default=5e-5, 
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of epochs for training")
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo", 
                        help="Model to use for training")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the conversation data
    data_path = Path(args.data_path).resolve()
    print(f"Loading data from {data_path}")
    training_data = load_convo_data(data_path)
    
    # Create a dataset from the training data
    dataset = Dataset.from_list(training_data)
    
    # Split dataset into train and eval
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Initialize the environment
    env = ObsidianAgentEnv(
        dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    
    # Configure GRPO training
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        use_vllm=True,  # GRPO requires vLLM
        temperature=0.7,
        top_p=0.9,
    )
    
    # Initialize the trainer
    trainer = GRPOEnvTrainer(
        model=args.model_name,
        env=env,
        reward_funcs=env.get_reward_funcs(),
        args=grpo_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    
    # Train the model
    trainer.train()
    
    # Print some evaluation metrics
    rewards = compute_all_rewards(
        completions=trainer.eval_completions,
        env=env,
        eval_dataset=dataset["test"],
    )
    
    print("Evaluation Rewards:")
    for i, reward in enumerate(rewards):
        print(f"Example {i+1}: {reward}")
    
    print(f"Average Reward: {sum(rewards) / len(rewards)}")
    
    # Save the final results
    results = {
        "rewards": rewards,
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0,
    }
    
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Training complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
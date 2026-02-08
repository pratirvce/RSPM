"""
Run RSPM evaluation on HaluMem dataset
Goal: Achieve >95% Temporal Consistency Score
"""
import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cookbook.memoryscope.data_loader import MemoryScopeDataset
from cookbook.memoryscope.metrics import MemoryScopeMetrics
from cookbook.memoryscope.rspm_agent import RSPMAgent

class HaluMemEvaluator:
    """Evaluate RSPM on HaluMem dataset"""
    
    def __init__(self, dataset_path: str, reme_url: str = "http://localhost:8002"):
        self.dataset_path = dataset_path
        self.reme_url = reme_url
        self.results = {
            "dataset": dataset_path,
            "start_time": datetime.now().isoformat(),
            "configurations": {},
            "results": {}
        }
    
    def run_evaluation(self, config_name: str, agent: RSPMAgent, limit: int = None):
        """Run evaluation with specific RSPM configuration"""
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {config_name}")
        print(f"{'='*60}")
        
        # Load dataset
        print(f"\nLoading dataset from {self.dataset_path}...")
        dataset = MemoryScopeDataset(self.dataset_path)
        
        total_conversations = len(dataset)
        if limit:
            total_conversations = min(total_conversations, limit)
            print(f"Limiting evaluation to {limit} conversations")
        
        print(f"Total conversations to evaluate: {total_conversations}")
        
        # Clear workspace
        print("\nClearing agent workspace...")
        agent.clear_workspace()
        
        # Initialize metrics
        metrics = MemoryScopeMetrics()
        
        # Process each conversation
        print(f"\nProcessing conversations...")
        for idx in range(total_conversations):
            conversation = dataset[idx]
            conv_id = conversation.get('conversation_id', f'conv_{idx}')
            
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"\nProgress: {idx + 1}/{total_conversations} conversations")
                print(f"  Current conversation: {conv_id}")
            
            try:
                # Process conversation with RSPM agent
                agent_response, conflicts = agent.process_conversation(conversation)
                
                # Evaluate response
                ground_truth = conversation.get('ground_truth', {})
                result = agent.evaluate_response(agent_response, ground_truth)
                
                # Update metrics
                metrics.update(result)
                
                # Log detailed results for first few conversations
                if idx < 5:
                    print(f"\n  Conversation {conv_id} results:")
                    print(f"    - Correct: {result['correct']}")
                    print(f"    - Conflicts detected: {len(conflicts)}")
                    print(f"    - Used outdated info: {result.get('used_outdated', False)}")
                
            except Exception as e:
                print(f"\n  ✗ Error processing conversation {conv_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Compute final metrics
        print(f"\n{'='*60}")
        print(f"Computing final metrics for {config_name}...")
        print(f"{'='*60}")
        
        final_metrics = metrics.compute()
        
        # Display results
        print(f"\n{config_name} Results:")
        print(f"  Temporal Consistency Score: {final_metrics['tcs']:.2%}")
        print(f"  Overall Accuracy: {final_metrics['overall_accuracy']:.2%}")
        print(f"  Memory Efficiency Ratio: {final_metrics['memory_efficiency_ratio']:.2f}")
        print(f"  Total Evaluated: {final_metrics['total']}")
        print(f"  Correct: {final_metrics['correct']}")
        print(f"  Incorrect: {final_metrics['incorrect']}")
        
        # Check if goal achieved
        if final_metrics['tcs'] >= 0.95:
            print(f"\n  🎉 GOAL ACHIEVED! TCS = {final_metrics['tcs']:.2%} >= 95%")
        else:
            print(f"\n  ⚠ Goal not reached. TCS = {final_metrics['tcs']:.2%} < 95%")
            print(f"     Gap: {(0.95 - final_metrics['tcs']) * 100:.2f}%")
        
        # Store results
        self.results["results"][config_name] = {
            "metrics": final_metrics,
            "goal_achieved": final_metrics['tcs'] >= 0.95
        }
        
        return final_metrics
    
    def run_all_configurations(self, limit: int = None):
        """Run evaluation with different RSPM configurations"""
        
        configurations = [
            {
                "name": "RSPM-Basic",
                "params": {
                    "workspace_id": "halumem_rspm_basic",
                    "sleep_frequency": 10,
                    "enable_hierarchical": False,
                    "enable_reranking": False,
                    "reme_url": self.reme_url
                }
            },
            {
                "name": "RSPM-Hierarchical",
                "params": {
                    "workspace_id": "halumem_rspm_hierarchical",
                    "sleep_frequency": 10,
                    "enable_hierarchical": True,
                    "enable_reranking": False,
                    "reme_url": self.reme_url
                }
            },
            {
                "name": "RSPM-Reranking",
                "params": {
                    "workspace_id": "halumem_rspm_reranking",
                    "sleep_frequency": 10,
                    "enable_hierarchical": False,
                    "enable_reranking": True,
                    "reme_url": self.reme_url
                }
            },
            {
                "name": "RSPM-Advanced",
                "params": {
                    "workspace_id": "halumem_rspm_advanced",
                    "sleep_frequency": 10,
                    "enable_hierarchical": True,
                    "enable_reranking": True,
                    "reme_url": self.reme_url
                }
            }
        ]
        
        print(f"\n{'='*60}")
        print("HaluMem RSPM Evaluation")
        print(f"{'='*60}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Configurations: {len(configurations)}")
        print(f"ReMe URL: {self.reme_url}")
        if limit:
            print(f"Conversation limit: {limit}")
        print(f"{'='*60}")
        
        for config in configurations:
            print(f"\n\n{'#'*60}")
            print(f"# Configuration: {config['name']}")
            print(f"{'#'*60}")
            
            # Store configuration
            self.results["configurations"][config['name']] = config['params']
            
            # Create agent with configuration
            agent = RSPMAgent(**config['params'])
            
            # Run evaluation
            try:
                self.run_evaluation(config['name'], agent, limit=limit)
            except Exception as e:
                print(f"\n✗ Error in {config['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final summary
        self.print_final_summary()
        
        # Save results
        self.save_results()
    
    def print_final_summary(self):
        """Print final comparison of all configurations"""
        
        print(f"\n\n{'='*60}")
        print("FINAL EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"\nDataset: {self.dataset_path}")
        
        if not self.results["results"]:
            print("\n⚠️  No results available. All configurations may have failed.")
            print("Check the logs above for error details.")
            return
        
        print(f"\n{'Configuration':<25} {'TCS':<10} {'Accuracy':<10} {'Goal':<10}")
        print("-" * 60)
        
        for config_name, result in self.results["results"].items():
            metrics = result["metrics"]
            goal = "✓ YES" if result["goal_achieved"] else "✗ NO"
            
            print(f"{config_name:<25} {metrics['tcs']:.2%}    {metrics['overall_accuracy']:.2%}    {goal}")
        
        # Best configuration
        if self.results["results"]:
            best_config = max(
                self.results["results"].items(),
                key=lambda x: x[1]["metrics"]["tcs"]
            )
            
            print(f"\n{'='*60}")
            print(f"BEST CONFIGURATION: {best_config[0]}")
            print(f"  TCS: {best_config[1]['metrics']['tcs']:.2%}")
            print(f"  Accuracy: {best_config[1]['metrics']['overall_accuracy']:.2%}")
            print(f"  Goal Achieved: {best_config[1]['goal_achieved']}")
            print(f"{'='*60}")
    
    def save_results(self):
        """Save evaluation results to JSON"""
        
        self.results["end_time"] = datetime.now().isoformat()
        
        # Create results directory
        results_dir = "results/halumem"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.basename(self.dataset_path).replace('.jsonl', '')
        filename = f"{results_dir}/{dataset_name}_{timestamp}.json"
        
        # Save results
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Run RSPM evaluation on HaluMem dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="medium",
        choices=["medium", "long"],
        help="Dataset version to evaluate (medium or long)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of conversations to evaluate (for testing)"
    )
    parser.add_argument(
        "--reme-url",
        type=str,
        default="http://localhost:8002",
        help="ReMe service URL"
    )
    
    args = parser.parse_args()
    
    # Construct dataset path
    dataset_path = f"datasets/memoryscope/halumem_{args.dataset}.jsonl"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset not found at {dataset_path}")
        print("\nPlease run the following commands first:")
        print("1. python cookbook/memoryscope/download_halumem.py")
        print("2. python cookbook/memoryscope/halumem_adapter.py")
        sys.exit(1)
    
    # Create evaluator
    evaluator = HaluMemEvaluator(
        dataset_path=dataset_path,
        reme_url=args.reme_url
    )
    
    # Run evaluation
    evaluator.run_all_configurations(limit=args.limit)

if __name__ == "__main__":
    main()

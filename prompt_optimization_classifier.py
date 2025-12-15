"""
Prompt Optimization Loop for PASS/FAIL Scooter Parking Classifier

This module implements an iterative prompt optimization system for classifying
scooter parking photos as PASS (in nest) or FAIL (not in nest).
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the optimization loop"""
    # Temperature options
    temperatures: List[float] = None
    
    # Precision threshold for FAIL class
    precision_threshold: float = 0.90
    
    # Maximum iterations
    max_iterations: int = 5
    
    # Filename keywords for label extraction
    fail_keyword: str = "expected_fail"
    pass_keyword: str = "expected_pass"
    
    # Train/test split
    dev_split: float = 0.7
    random_seed: int = 42
    
    # LLM provider (to be set by user)
    llm_provider: str = "openai"  # or "anthropic", "openrouter"
    
    def __post_init__(self):
        if self.temperatures is None:
            self.temperatures = [0.1, 0.2, 0.3]


@dataclass
class EvaluationResult:
    """Result of evaluating a single image"""
    decision: str  # "PASS" or "FAIL"
    reason: str
    raw_output: str
    parse_success: bool = True


@dataclass
class Metrics:
    """Classification metrics"""
    recall_fail: float
    precision_fail: float
    f2_fail: float
    tp: int  # True Positives (FAIL predicted as FAIL)
    fp: int  # False Positives (PASS predicted as FAIL)
    fn: int  # False Negatives (FAIL predicted as PASS)
    tn: int  # True Negatives (PASS predicted as PASS)
    temperature: float
    prompt_version: int = 0
    
    def __str__(self):
        return (
            f"Metrics (temp={self.temperature}, v={self.prompt_version}): "
            f"Recall={self.recall_fail:.3f}, Precision={self.precision_fail:.3f}, "
            f"F2={self.f2_fail:.3f}, TP={self.tp}, FP={self.fp}, FN={self.fn}, TN={self.tn}"
        )


class LLMClient:
    """
    Abstract LLM client interface.
    Users should subclass this and implement call_llm method.
    """
    
    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        model: str = None,
        image_path: str = None
    ) -> str:
        """
        Call the LLM with the given prompts.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt (may include image reference)
            temperature: Temperature setting
            model: Model name (optional, provider-specific)
            image_path: Path to image (for vision models)
            
        Returns:
            Raw text response from LLM
        """
        raise NotImplementedError("Subclass must implement call_llm")
    
    def call_vision_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str,
        temperature: float,
        model: str = None
    ) -> str:
        """
        Call vision LLM with image.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            image_path: Path to image file or URL
            temperature: Temperature setting
            model: Model name (optional)
            
        Returns:
            Raw text response from LLM
        """
        # Default implementation combines user prompt with image
        combined_prompt = f"{user_prompt}\n\nImage: {image_path}"
        return self.call_llm(system_prompt, combined_prompt, temperature, model, image_path)


class PromptOptimizer:
    """Main class for prompt optimization loop"""
    
    def __init__(self, config: Config, llm_client: LLMClient, 
                 initial_system_prompt: Optional[str] = None,
                 initial_user_prompt: Optional[str] = None):
        self.config = config
        self.llm_client = llm_client
        self.dev_data = None
        self.test_data = None
        self.best_metrics = None
        self.best_prompt_system = None
        self.best_prompt_user = None
        self.best_temperature = None
        self.iteration_history = []
        self.initial_system_prompt = initial_system_prompt
        self.initial_user_prompt = initial_user_prompt
        
    def load_and_prepare_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load CSV and add label column based on filename.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with image_path and label columns
        """
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Extract label from filename
        def extract_label(path: str) -> str:
            path_lower = str(path).lower()
            if self.config.fail_keyword.lower() in path_lower:
                return "FAIL"
            elif self.config.pass_keyword.lower() in path_lower:
                return "PASS"
            else:
                logger.warning(f"Could not determine label for {path}")
                return None
        
        df['label'] = df['image_path'].apply(extract_label)
        df = df[df['label'].notna()].copy()
        
        logger.info(f"Loaded {len(df)} images: {df['label'].value_counts().to_dict()}")
        return df
    
    def split_data(self, df: pd.DataFrame):
        """Split data into dev and test sets"""
        np.random.seed(self.config.random_seed)
        indices = np.random.permutation(len(df))
        split_idx = int(len(df) * self.config.dev_split)
        
        dev_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        self.dev_data = df.iloc[dev_indices].copy().reset_index(drop=True)
        self.test_data = df.iloc[test_indices].copy().reset_index(drop=True)
        
        logger.info(f"Split: {len(self.dev_data)} dev, {len(self.test_data)} test")
        logger.info(f"Dev labels: {self.dev_data['label'].value_counts().to_dict()}")
        logger.info(f"Test labels: {self.test_data['label'].value_counts().to_dict()}")
    
    def get_initial_prompts(self) -> Tuple[str, str]:
        """Generate initial system and user prompts"""
        # Use custom prompts if provided, otherwise use defaults
        if self.initial_system_prompt and self.initial_user_prompt:
            return self.initial_system_prompt, self.initial_user_prompt
        
        # Default prompts
        system_prompt = """You are an expert at analyzing scooter parking photos. Your task is to determine if a scooter is correctly parked in a designated nest/corral.

A valid nest/corral is identified by:
1. Designated micromobility parking symbols (white P-in-circle, scooter icon, or bicycle icon)
2. Black-and-white posts or bollards forming a corral boundary
3. White painted boxes or boundary lines marking the parking area
4. Any combination of the above elements

CRITICAL RULE: If the scooter is NOT clearly inside a valid nest/corral → the decision must be FAIL.

You must output your analysis as strict JSON with this exact format:
{
    "decision": "PASS" or "FAIL",
    "reason": "Brief explanation of your decision"
}"""

        user_prompt = """Analyze this scooter parking photo and determine if the scooter is correctly parked.

Step-by-step analysis:
1. Identify if there is a valid nest/corral visible in the image
2. Determine if the scooter is clearly positioned inside the nest/corral boundaries
3. If the scooter is outside the nest, partially outside, or no nest is visible → FAIL
4. If the scooter is clearly and fully inside a valid nest → PASS

Output your decision as JSON with "decision" and "reason" fields."""

        return system_prompt, user_prompt
    
    def evaluate_single_image(
        self,
        image_path: str,
        prompt_system: str,
        prompt_user: str,
        temperature: float
    ) -> EvaluationResult:
        """
        Evaluate a single image with the given prompts and temperature.
        
        Returns:
            EvaluationResult with decision and metadata
        """
        # Ensure prompts are strings, never None
        if prompt_system is None:
            prompt_system = ""
        if prompt_user is None:
            prompt_user = ""
        if temperature is None:
            temperature = 0.1
        
        try:
            raw_output = self.llm_client.call_vision_llm(
                system_prompt=prompt_system,
                user_prompt=prompt_user,
                image_path=image_path,
                temperature=float(temperature)
            )
            
            # Parse JSON from response
            # Try to extract JSON if it's embedded in text
            json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = raw_output.strip()
            
            # Remove markdown code blocks if present
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*', '', json_str)
            json_str = json_str.strip()
            
            parsed = json.loads(json_str)
            decision = parsed.get("decision", "").upper()
            reason = parsed.get("reason", "")
            
            if decision not in ["PASS", "FAIL"]:
                logger.warning(f"Invalid decision '{decision}' for {image_path}")
                decision = "FAIL"  # Default to FAIL for invalid output
                parse_success = False
            else:
                parse_success = True
                
            return EvaluationResult(
                decision=decision,
                reason=reason,
                raw_output=raw_output,
                parse_success=parse_success
            )
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error for {image_path}: {e}")
            return EvaluationResult(
                decision="FAIL",  # Default to FAIL on parse error
                reason=f"Parse error: {str(e)}",
                raw_output=raw_output if 'raw_output' in locals() else "",
                parse_success=False
            )
        except Exception as e:
            logger.error(f"Error evaluating {image_path}: {e}")
            # Return a special error result that will be handled in evaluate_on_dataset
            return EvaluationResult(
                decision="__ERROR__",  # Special marker for API errors
                reason=f"API Error: {str(e)}",
                raw_output="",
                parse_success=False
            )
    
    def compute_metrics(
        self,
        predictions: List[str],
        ground_truth: List[str],
        temperature: float,
        prompt_version: int = 0
    ) -> Metrics:
        """
        Compute classification metrics (FAIL is positive class).
        
        Args:
            predictions: List of predicted labels ("PASS" or "FAIL")
            ground_truth: List of true labels
            temperature: Temperature used
            prompt_version: Version number of prompt
            
        Returns:
            Metrics object
        """
        # FAIL is positive class
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == "FAIL" and g == "FAIL")
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == "FAIL" and g == "PASS")
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p == "PASS" and g == "FAIL")
        tn = sum(1 for p, g in zip(predictions, ground_truth) if p == "PASS" and g == "PASS")
        
        recall_fail = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_fail = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # F2 score: recall-weighted (beta=2)
        f2_fail = (5 * precision_fail * recall_fail) / (4 * precision_fail + recall_fail) if (4 * precision_fail + recall_fail) > 0 else 0.0
        
        return Metrics(
            recall_fail=recall_fail,
            precision_fail=precision_fail,
            f2_fail=f2_fail,
            tp=tp, fp=fp, fn=fn, tn=tn,
            temperature=temperature,
            prompt_version=prompt_version
        )
    
    def evaluate_on_dataset(
        self,
        data: pd.DataFrame,
        prompt_system: str,
        prompt_user: str,
        temperature: float,
        prompt_version: int = 0
    ) -> Tuple[Metrics, List[Dict]]:
        """
        Evaluate prompts on a dataset.
        
        Returns:
            Tuple of (Metrics, list of misclassified examples)
        """
        predictions = []
        misclassified = []
        errors = []  # Track API errors
        successful_calls = 0
        failed_calls = 0
        
        # Ensure temperature is float
        if temperature is None:
            temperature = 0.1
        temperature = float(temperature)
        
        logger.info(f"Evaluating on {len(data)} images (temp={temperature})...")
        logger.info(f"System Prompt: {prompt_system[:100] if prompt_system else 'None'}...")
        logger.info(f"User Prompt: {prompt_user[:100] if prompt_user else 'None'}...")
        
        for idx, row in data.iterrows():
            result = self.evaluate_single_image(
                row['image_path'],
                prompt_system,
                prompt_user,
                temperature
            )
            
            # Handle API errors (Option A: strict - count as wrong prediction)
            if result.decision == "__ERROR__":
                failed_calls += 1
                errors.append({
                    'image_path': row['image_path'],
                    'ground_truth': row['label'],
                    'error': result.reason
                })
                # Count error as wrong prediction:
                # If true_label == "FAIL" → FN (we failed to catch it)
                # If true_label == "PASS" → FP (we incorrectly flagged it)
                if row['label'] == "FAIL":
                    predictions.append("PASS")  # Count as FN
                else:
                    predictions.append("FAIL")  # Count as FP
            else:
                successful_calls += 1
                predictions.append(result.decision)
            
            # Track misclassifications (only for successful calls)
            if result.decision != "__ERROR__" and result.decision != row['label']:
                misclassified.append({
                    'image_path': row['image_path'],
                    'ground_truth': row['label'],
                    'model_decision': result.decision,
                    'reason': result.reason,
                    'parse_success': result.parse_success
                })
        
        metrics = self.compute_metrics(
            predictions,
            data['label'].tolist(),
            temperature,
            prompt_version
        )
        
        # Log evaluation summary
        logger.info(f"\nEvaluation Summary:")
        logger.info(f"  Total images: {len(data)}")
        logger.info(f"  Successful calls: {successful_calls}")
        logger.info(f"  Failed calls: {failed_calls}")
        if failed_calls > 0:
            logger.warning(f"  ⚠️  {failed_calls} API calls failed - counted as wrong predictions (Option A: strict)")
        
        return metrics, misclassified
    
    def select_best_configuration(
        self,
        results: List[Tuple[Metrics, str, str, float]]
    ) -> Tuple[Metrics, str, str, float]:
        """
        Select best (prompt, temperature) combo.
        
        Criteria:
        1. Maximize recall_fail
        2. Subject to precision_fail >= threshold
        3. If tied, use F2 score
        
        Args:
            results: List of (Metrics, prompt_system, prompt_user, temperature)
            
        Returns:
            Best (Metrics, prompt_system, prompt_user, temperature)
        """
        # Filter by precision threshold
        valid_results = [
            r for r in results
            if r[0].precision_fail >= self.config.precision_threshold
        ]
        
        if not valid_results:
            logger.warning("No results meet precision threshold, using best available")
            valid_results = results
        
        # Sort by recall (primary), then F2 (tie-breaker)
        valid_results.sort(key=lambda x: (x[0].recall_fail, x[0].f2_fail), reverse=True)
        
        return valid_results[0]
    
    def optimize_prompt(
        self,
        current_system: str,
        current_user: str,
        current_metrics: Metrics,
        misclassified: List[Dict],
        temperature_strategy: str,
        all_temperature_results: List[Tuple[Metrics, float]] = None
    ) -> Dict:
        """
        Call optimizer LLM to improve prompts.
        
        Returns:
            Dict with new prompts and temperature strategy
        """
        # Prepare misclassified examples for optimizer
        misclassified_text = "\n\n".join([
            f"Image: {ex['image_path']}\n"
            f"Ground Truth: {ex['ground_truth']}\n"
            f"Model Decision: {ex['model_decision']}\n"
            f"Model Reason: {ex.get('reason', 'N/A')}"
            for ex in misclassified[:10]  # Limit to 10 examples
        ])
        
        optimizer_system = """You are an expert at optimizing prompts for image classification tasks. Your goal is to improve prompt clarity and effectiveness based on classification errors.

You must output strict JSON with this exact format:
{
    "system_prompt": "...",
    "user_prompt": "...",
    "temperature_strategy": "reuse_best" or "full_sweep" or "subset" or "fixed"
}

Guidelines:
- Make SMALL, targeted improvements only
- Preserve the JSON output format requirement
- Focus on improving clarity of nest identification rules
- Address specific error patterns shown in misclassified examples
- Keep changes minimal and incremental
- If temperature is fixed, use "fixed" for temperature_strategy"""

        # Format temperature results if available
        temp_results_text = ""
        if all_temperature_results:
            temp_results_text = "\nTemperature Results:\n"
            for metrics, temp in all_temperature_results:
                temp_results_text += (
                    f"  Temp {temp}: Recall={metrics.recall_fail:.3f}, "
                    f"Precision={metrics.precision_fail:.3f}, F2={metrics.f2_fail:.3f}\n"
                )
            temp_results_text += "\n"
        
        # Build optimizer user prompt
        if all_temperature_results is None or temperature_strategy == "fixed":
            # Temperature is fixed - don't include temperature strategy info
            optimizer_user = f"""Current System Prompt:
{current_system}

Current User Prompt:
{current_user}

Current Performance (temperature is fixed at {current_metrics.temperature}):
- Recall (FAIL): {current_metrics.recall_fail:.3f}
- Precision (FAIL): {current_metrics.precision_fail:.3f}
- F2 Score: {current_metrics.f2_fail:.3f}
- TP={current_metrics.tp}, FP={current_metrics.fp}, FN={current_metrics.fn}, TN={current_metrics.tn}

Misclassified Examples:
{misclassified_text}

Analyze the errors and suggest improved prompts. Make small, targeted changes to:
1. Improve clarity of nest identification criteria
2. Strengthen the FAIL rule (scooter NOT in nest → FAIL)
3. Address specific failure patterns

Note: Temperature is fixed and will not be optimized. Focus only on improving the prompts.

Output the improved prompts as JSON with this format:
{{
    "system_prompt": "...",
    "user_prompt": "...",
    "temperature_strategy": "fixed"
}}"""
        else:
            # Original temperature optimization logic (kept for backward compatibility)
            optimizer_user = f"""Current System Prompt:
{current_system}

Current User Prompt:
{current_user}

Current Best Performance (temp={current_metrics.temperature}):
- Recall (FAIL): {current_metrics.recall_fail:.3f}
- Precision (FAIL): {current_metrics.precision_fail:.3f}
- F2 Score: {current_metrics.f2_fail:.3f}
- TP={current_metrics.tp}, FP={current_metrics.fp}, FN={current_metrics.fn}, TN={current_metrics.tn}
{temp_results_text}
Temperature Strategy Used: {temperature_strategy}

Misclassified Examples:
{misclassified_text}

Analyze the errors and suggest improved prompts. Make small, targeted changes to:
1. Improve clarity of nest identification criteria
2. Strengthen the FAIL rule (scooter NOT in nest → FAIL)
3. Address specific failure patterns

For temperature_strategy, choose:
- "reuse_best": If current temperature is clearly best
- "full_sweep": If you want to retest all temperatures
- "subset": If you want to test best and adjacent temperatures

Output the improved prompts as JSON."""

        try:
            raw_output = self.llm_client.call_llm(
                system_prompt=optimizer_system,
                user_prompt=optimizer_user,
                temperature=0.3  # Lower temperature for optimization
            )
            
            # Parse JSON
            json_match = re.search(r'\{[^{}]*"system_prompt"[^{}]*\}', raw_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = raw_output.strip()
            
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```\s*', '', json_str)
            json_str = json_str.strip()
            
            result = json.loads(json_str)
            
            # Validate
            if "system_prompt" not in result or "user_prompt" not in result:
                raise ValueError("Missing required fields in optimizer output")
            
            # Handle temperature strategy (may be "fixed" if temperature is not optimized)
            if result.get("temperature_strategy") not in ["reuse_best", "full_sweep", "subset", "fixed"]:
                result["temperature_strategy"] = "fixed" if temperature_strategy == "fixed" else "reuse_best"
            
            # Check if prompts actually changed
            system_changed = result["system_prompt"] != current_system
            user_changed = result["user_prompt"] != current_user
            
            if not system_changed and not user_changed:
                logger.warning("⚠️  Optimizer returned identical prompts - no changes made")
            else:
                if system_changed:
                    logger.info(f"✓ System prompt updated (length: {len(current_system)} → {len(result['system_prompt'])})")
                if user_changed:
                    logger.info(f"✓ User prompt updated (length: {len(current_user)} → {len(result['user_prompt'])})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prompt optimization: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            logger.warning("⚠️  Returning original prompts due to error")
            return {
                "system_prompt": current_system,
                "user_prompt": current_user,
                "temperature_strategy": "reuse_best"
            }
    
    def run_optimization_loop(self):
        """Main optimization loop"""
        logger.info("=" * 80)
        logger.info("Starting Prompt Optimization Loop")
        logger.info("=" * 80)
        
        # Initialize prompts
        prompt_system, prompt_user = self.get_initial_prompts()
        best_temperature = float(self.config.temperatures[0])
        
        # Ensure initial prompts are set as fallback
        if prompt_system is None:
            prompt_system = ""
        if prompt_user is None:
            prompt_user = ""
        
        # Set initial best configuration as fallback
        self.best_prompt_system = prompt_system
        self.best_prompt_user = prompt_user
        self.best_temperature = best_temperature
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"\n{'='*80}")
            logger.info(f"Iteration {iteration} - Prompt Optimization Only")
            logger.info(f"{'='*80}")
            
            # Use fixed temperature (no temperature optimization)
            fixed_temp = float(self.config.temperatures[0])
            logger.info(f"Temperature: {fixed_temp} (fixed - not optimized)")
            
            # Evaluate current prompts with fixed temperature
            metrics, misclassified = self.evaluate_on_dataset(
                self.dev_data,
                prompt_system,
                prompt_user,
                fixed_temp,
                prompt_version=iteration
            )
            
            logger.info(f"\n{metrics}")
            
            # Current configuration
            best_metrics = metrics
            best_system = prompt_system
            best_user = prompt_user
            best_temp = fixed_temp
            
            # Check for improvement
            if self.best_metrics is None:
                improvement = True
            else:
                recall_improvement = best_metrics.recall_fail - self.best_metrics.recall_fail
                f2_improvement = best_metrics.f2_fail - self.best_metrics.f2_fail
                improvement = (recall_improvement > 0.01) or (f2_improvement > 0.01)
            
            # Update best if improved
            if self.best_metrics is None or improvement:
                self.best_metrics = best_metrics
                self.best_prompt_system = best_system
                self.best_prompt_user = best_user
                self.best_temperature = best_temp
                logger.info("✓ New best configuration!")
            else:
                logger.info("No significant improvement")
            
            # Gather misclassified examples for optimizer
            _, misclassified = self.evaluate_on_dataset(
                self.dev_data,
                best_system,
                best_user,
                best_temp,
                prompt_version=iteration
            )
            
            # Store iteration history
            self.iteration_history.append({
                "iteration": iteration,
                "metrics": best_metrics,
                "temperature_strategy": "fixed",
                "misclassified_count": len(misclassified)
            })
            
            # Optimize prompts (except on last iteration)
            if iteration < self.config.max_iterations - 1:
                logger.info("\nOptimizing prompts...")
                logger.info(f"Current prompts before optimization:")
                logger.info(f"  System: {len(best_system)} chars")
                logger.info(f"  User: {len(best_user)} chars")
                
                # Optimize prompts (temperature is fixed, so no temperature strategy needed)
                optimized = self.optimize_prompt(
                    best_system,
                    best_user,
                    best_metrics,
                    misclassified,
                    temperature_strategy="fixed",  # Not used when temperature is fixed
                    all_temperature_results=None  # Not needed when temperature is fixed
                )
                
                # Check if prompts actually changed
                old_system = prompt_system
                old_user = prompt_user
                prompt_system = optimized["system_prompt"]
                prompt_user = optimized["user_prompt"]
                
                if prompt_system == old_system and prompt_user == old_user:
                    logger.warning("⚠️  Prompts unchanged after optimization")
                else:
                    logger.info("✓ Prompts updated by optimizer")
                    if prompt_system != old_system:
                        logger.info(f"  System prompt changed: {len(old_system)} → {len(prompt_system)} chars")
                    if prompt_user != old_user:
                        logger.info(f"  User prompt changed: {len(old_user)} → {len(prompt_user)} chars")
            
            # Early stopping if no improvement
            if not improvement and iteration > 0:
                logger.info("Stopping early: no significant improvement")
                break
        
        logger.info(f"\n{'='*80}")
        logger.info("Optimization Complete")
        logger.info(f"{'='*80}")
        
        # Ensure best configuration is always set (fallback to initial if needed)
        if self.best_prompt_system is None:
            self.best_prompt_system = prompt_system
        if self.best_prompt_user is None:
            self.best_prompt_user = prompt_user
        if self.best_temperature is None:
            self.best_temperature = float(self.config.temperatures[0])
        
        # Ensure temperature is float
        self.best_temperature = float(self.best_temperature)
        
        logger.info(f"Final Best Configuration:")
        logger.info(f"  Temperature: {self.best_temperature}")
        logger.info(f"  System Prompt: {self.best_prompt_system[:100] if self.best_prompt_system else 'None'}...")
        logger.info(f"  User Prompt: {self.best_prompt_user[:100] if self.best_prompt_user else 'None'}...")
        if self.best_metrics:
            logger.info(f"  {self.best_metrics}")
        else:
            logger.warning("  No metrics available - using initial prompts")
    
    def evaluate_on_test_set(self):
        """Evaluate final best configuration on test set"""
        logger.info(f"\n{'='*80}")
        logger.info("Final Evaluation on Test Set")
        logger.info(f"{'='*80}")
        
        # Ensure best configuration is set (fallback if optimization didn't run)
        if self.best_prompt_system is None or self.best_prompt_user is None:
            logger.warning("Best prompts not set, using initial prompts")
            prompt_system, prompt_user = self.get_initial_prompts()
            self.best_prompt_system = prompt_system if self.best_prompt_system is None else self.best_prompt_system
            self.best_prompt_user = prompt_user if self.best_prompt_user is None else self.best_prompt_user
        
        if self.best_temperature is None:
            self.best_temperature = float(self.config.temperatures[0])
            logger.warning(f"Best temperature not set, using default: {self.best_temperature}")
        
        # Ensure temperature is float
        self.best_temperature = float(self.best_temperature)
        
        # Ensure prompts are strings
        if self.best_prompt_system is None:
            self.best_prompt_system = ""
        if self.best_prompt_user is None:
            self.best_prompt_user = ""
        
        logger.info(f"Using configuration:")
        logger.info(f"  Temperature: {self.best_temperature} (type: {type(self.best_temperature).__name__})")
        logger.info(f"  System Prompt length: {len(self.best_prompt_system)} chars")
        logger.info(f"  User Prompt length: {len(self.best_prompt_user)} chars")
        
        metrics, misclassified = self.evaluate_on_dataset(
            self.test_data,
            self.best_prompt_system,
            self.best_prompt_user,
            self.best_temperature,
            prompt_version=-1
        )
        
        logger.info(f"\nTest Set Results:")
        logger.info(f"  Recall (FAIL): {metrics.recall_fail:.3f}")
        logger.info(f"  Precision (FAIL): {metrics.precision_fail:.3f}")
        logger.info(f"  F2 Score: {metrics.f2_fail:.3f}")
        logger.info(f"\nConfusion Matrix (FAIL=positive):")
        logger.info(f"  TP={metrics.tp}, FP={metrics.fp}")
        logger.info(f"  FN={metrics.fn}, TN={metrics.tn}")
        
        logger.info(f"\nFinal Prompts:")
        logger.info(f"System Prompt:\n{self.best_prompt_system}")
        logger.info(f"\nUser Prompt:\n{self.best_prompt_user}")
        logger.info(f"\nTemperature: {self.best_temperature}")
        
        return metrics, misclassified


# Example usage and LLM client implementation
if __name__ == "__main__":
    # Example: OpenAI client implementation
    # Users should implement their own LLM client
    
    class OpenAILLMClient(LLMClient):
        """Example OpenAI client implementation"""
        
        def __init__(self, api_key: str, model: str = "gpt-4o"):
            self.api_key = api_key
            self.model = model
        
        def call_llm(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            model: str = None,
            image_path: str = None
        ) -> str:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        
        def call_vision_llm(
            self,
            system_prompt: str,
            user_prompt: str,
            image_path: str,
            temperature: float,
            model: str = None
        ) -> str:
            import openai
            from pathlib import Path
            
            client = openai.OpenAI(api_key=self.api_key)
            
            # Prepare image
            if Path(image_path).exists():
                # Local file - encode as base64
                import base64
                with open(image_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{image_data}"
            else:
                # Assume URL
                image_url = image_path
            
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
            
            response = client.chat.completions.create(
                model=model or "gpt-4o",
                messages=messages,
                temperature=temperature,
                max_tokens=500
            )
            
            return response.choices[0].message.content
    
    # Example usage
    print("This is a library module. Import and use as follows:")
    print("""
    from prompt_optimization_classifier import PromptOptimizer, Config, LLMClient
    
    # 1. Create your LLM client (implement LLMClient interface)
    class MyLLMClient(LLMClient):
        def call_llm(self, system_prompt, user_prompt, temperature, model=None, image_path=None):
            # Your implementation
            pass
        
        def call_vision_llm(self, system_prompt, user_prompt, image_path, temperature, model=None):
            # Your implementation
            pass
    
    # 2. Configure
    config = Config(
        temperatures=[0.1, 0.2, 0.3],
        precision_threshold=0.90,
        max_iterations=5
    )
    
    # 3. Run optimization
    optimizer = PromptOptimizer(config, MyLLMClient())
    optimizer.load_and_prepare_data("your_data.csv")
    optimizer.split_data(optimizer.load_and_prepare_data("your_data.csv"))
    optimizer.run_optimization_loop()
    optimizer.evaluate_on_test_set()
    """)


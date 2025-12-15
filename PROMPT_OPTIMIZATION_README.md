# Prompt Optimization Loop for Scooter Parking Classifier

This system implements an iterative prompt optimization loop for classifying scooter parking photos as PASS (in nest) or FAIL (not in nest).

## Features

- **Iterative Prompt Optimization**: Uses an LLM to iteratively improve prompts based on classification errors
- **Temperature Optimization**: Tests multiple temperatures and adaptively chooses which to retest
- **Recall-Focused**: Maximizes recall for FAIL class (primary goal) while maintaining precision (secondary goal)
- **F2 Score Optimization**: Uses F2 score (recall-weighted) for tie-breaking
- **Strict JSON Parsing**: Enforces JSON output format with fallback handling
- **Modular Design**: Easy to plug in different LLM providers

## Requirements

```bash
pip install pandas numpy openai anthropic python-dotenv
```

## Quick Start

### 1. Prepare Your Data

Create a CSV file with at least one column:
- `image_path`: Local file path or URL to the image

The ground-truth label is extracted from the filename:
- Filenames containing `expected_fail` → label = "FAIL"
- Filenames containing `expected_pass` → label = "PASS"

Example CSV:
```csv
image_path
/path/to/photo_expected_fail_1.jpg
/path/to/photo_expected_pass_1.jpg
https://example.com/photo_expected_fail_2.jpg
```

### 2. Configure

Edit the notebook `scooter_parking_prompt_optimization.ipynb`:

1. Set your API key:
   ```python
   API_KEY = "your-api-key-here"
   ```

2. Choose your LLM provider:
   ```python
   PROVIDER = "openai"  # or "anthropic", "openrouter"
   ```

3. Update CSV path:
   ```python
   CSV_PATH = "path/to/your/data.csv"
   ```

4. Adjust configuration (optional):
   ```python
   config = Config(
       temperatures=[0.1, 0.2, 0.3],
       precision_threshold=0.90,
       max_iterations=5,
       fail_keyword="expected_fail",
       pass_keyword="expected_pass"
   )
   ```

### 3. Run

Execute the notebook cells in order. The system will:
1. Load and split your data (70% dev, 30% test)
2. Run the optimization loop (5 iterations by default)
3. Evaluate on the held-out test set
4. Display final metrics and optimized prompts

## How It Works

### Optimization Loop

1. **Iteration 0**: Tests all temperatures [0.1, 0.2, 0.3]
2. **Iterations 1+**: 
   - Optimizer LLM decides temperature strategy:
     - `reuse_best`: Only test the best temperature
     - `full_sweep`: Test all temperatures
     - `subset`: Test best and adjacent temperatures
   - Evaluates prompts on dev set
   - Selects best configuration (max recall, precision >= threshold, F2 tie-breaker)
   - Gathers misclassified examples
   - Calls optimizer LLM to improve prompts

### Selection Criteria

The best configuration is selected by:
1. **Primary**: Maximize recall for FAIL class
2. **Constraint**: Precision for FAIL >= threshold (default 0.90)
3. **Tie-breaker**: F2 score (recall-weighted)

### Prompt Structure

The system enforces strict JSON output:
```json
{
    "decision": "PASS" or "FAIL",
    "reason": "Brief explanation"
}
```

Invalid JSON or parse errors default to FAIL decision.

## Configuration Options

```python
Config(
    temperatures=[0.1, 0.2, 0.3],      # Temperatures to test
    precision_threshold=0.90,           # Minimum precision for FAIL
    max_iterations=5,                   # Max optimization iterations
    fail_keyword="expected_fail",        # Filename keyword for FAIL
    pass_keyword="expected_pass",       # Filename keyword for PASS
    dev_split=0.7,                       # Dev/test split ratio
    random_seed=42                       # Random seed for reproducibility
)
```

## LLM Provider Support

### OpenAI (GPT-4 Vision)
```python
llm_client = OpenAILLMClient(api_key=API_KEY, model="gpt-4o")
```

### Anthropic (Claude)
```python
llm_client = AnthropicLLMClient(api_key=API_KEY)
```

### OpenRouter
```python
llm_client = OpenRouterLLMClient(api_key=API_KEY, model="openai/gpt-4o")
```

### Custom Provider

Implement the `LLMClient` interface:

```python
class MyLLMClient(LLMClient):
    def call_llm(self, system_prompt, user_prompt, temperature, model=None, image_path=None):
        # Your implementation
        return response_text
    
    def call_vision_llm(self, system_prompt, user_prompt, image_path, temperature, model=None):
        # Your implementation for vision models
        return response_text
```

## Output

The system provides:

1. **Per-iteration metrics**:
   - Recall, Precision, F2 for FAIL class
   - Confusion matrix (TP/FP/FN/TN)
   - Temperature strategy used

2. **Final test set results**:
   - Final metrics on held-out test set
   - Misclassification analysis
   - Optimized prompts

3. **Optimized prompts**:
   - System prompt
   - User prompt
   - Best temperature

## Example Output

```
Iteration 0
Temperature strategy: full_sweep
Testing temperatures: [0.1, 0.2, 0.3]

Metrics (temp=0.1, v=0): Recall=0.850, Precision=0.920, F2=0.865, TP=85, FP=7, FN=15, TN=93
Metrics (temp=0.2, v=0): Recall=0.870, Precision=0.915, F2=0.880, TP=87, FP=8, FN=13, TN=92
Metrics (temp=0.3, v=0): Recall=0.860, Precision=0.910, F2=0.870, TP=86, FP=9, FN=14, TN=91

Best configuration:
  Temperature: 0.2
  Metrics: Recall=0.870, Precision=0.915, F2=0.880
✓ New best configuration!
```

## Troubleshooting

### JSON Parse Errors
- The system defaults to FAIL on parse errors
- Check LLM output format matches expected JSON
- Adjust prompts if parsing fails frequently

### Low Recall
- Optimizer should focus on strengthening FAIL rules
- Review false negatives (missed FAIL cases)
- Consider adjusting precision threshold

### Low Precision
- Optimizer should clarify nest identification criteria
- Review false positives (PASS labeled as FAIL)
- May need to raise precision threshold

## Files

- `prompt_optimization_classifier.py`: Core optimization module
- `scooter_parking_prompt_optimization.ipynb`: Usage notebook with examples
- `PROMPT_OPTIMIZATION_README.md`: This file

## License

Use as needed for your project.


# Deploying to Hugging Face Spaces

## Step 1: Install Gradio CLI (if not already installed)
```bash
pip install gradio
```

## Step 2: Authenticate with Hugging Face
```bash
huggingface-cli login
```
Or set your token:
```bash
export HF_TOKEN=your_huggingface_token_here
```

## Step 3: Deploy
From this directory, run:
```bash
gradio deploy
```

The command will:
1. Ask you to name your Space (e.g., "prompt-optimizer")
2. Create a new Space on Hugging Face
3. Upload your code
4. Deploy it permanently

## Alternative: Manual Deployment

If `gradio deploy` doesn't work, you can:
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose "Gradio" as the SDK
4. Upload your `app.py` file (extract from notebook)
5. Add `requirements.txt`
6. Push to the Space

## Note about API Keys

Make sure to set `OPENROUTER_API_KEY` as a secret in your Hugging Face Space:
1. Go to your Space settings
2. Add a secret named `OPENROUTER_API_KEY`
3. Enter your API key value

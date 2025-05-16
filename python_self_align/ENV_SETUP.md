# Environment Setup for Python Self-Align

This guide explains how to set up the environment variables for the Python Self-Align pipeline.

## Environment Variables

The Python Self-Align pipeline uses environment variables to configure its behavior. These variables can be set in a `.env` file at the root of the project, or they can be set as environment variables in your shell.

### Required Variables

- `HUGGINGFACE_API_KEY`: Your Hugging Face API key, used for accessing models from the Hugging Face Hub.
- `OPENAI_API_KEY`: Your OpenAI API key, used if you choose to use OpenAI models for generation.

### Optional Variables

- `MODEL`: The model to use for generation. Default: `"bigcode/starcoder2-15b"`. For testing with smaller models, you can use `"bigcode/starcoder2-3b"`.
- `MAX_WORKERS`: The number of worker processes to use for parallel processing. Default: `4`.
- `MAX_ITEMS`: The maximum number of samples to process in each pipeline stage. Default: `100`.

## Setting Up the .env File

1. Create a file named `.env` in the root of the project with the following content:

```
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
MODEL=bigcode/starcoder2-3b
MAX_WORKERS=4
MAX_ITEMS=10
```

2. Replace `your_huggingface_api_key_here` with your actual Hugging Face API key, which you can obtain from [Hugging Face Settings](https://huggingface.co/settings/tokens).

3. Replace `your_openai_api_key_here` with your actual OpenAI API key, which you can obtain from [OpenAI API Keys](https://platform.openai.com/api-keys).

## Getting API Keys

### Hugging Face

1. Create an account or sign in at [Hugging Face](https://huggingface.co/join).
2. Go to your profile settings by clicking on your profile picture in the top right corner and selecting "Settings".
3. Click on "Access Tokens" in the left sidebar.
4. Click "New token", provide a name for your token, and select the permissions you need (usually "read" is sufficient).
5. Click "Generate token" and copy the generated token.

### OpenAI

1. Create an account or sign in at [OpenAI](https://platform.openai.com/signup).
2. Go to [API Keys](https://platform.openai.com/api-keys) in your dashboard.
3. Click "Create new secret key", provide a name for your key, and click "Create secret key".
4. Copy the generated key immediately, as you won't be able to see it again.

## Testing Your Environment

You can test your environment setup with the following command:

```bash
python test_api_setup.py
```

This script will check if your API keys are working correctly and if the environment variables are loaded properly.

## Running the Pipeline

With your environment set up, you can run the pipeline using the modified script:

```bash
python run_pipeline_with_env.py --output_dir pipeline_output
```

This will use the settings from your `.env` file. You can override these settings with command-line arguments:

```bash
python run_pipeline_with_env.py --output_dir pipeline_output --max_samples 20 --num_workers 8
```

## Troubleshooting

If you encounter issues with the environment setup:

1. Make sure your `.env` file is in the root directory of the project.
2. Check that your API keys are correct and have not expired.
3. Try running `python test_api_setup.py` to diagnose the issue.
4. Ensure that you have all the required packages installed by running `pip install -r requirements.txt`.

For more information, please refer to the main README.md file in the project root. 
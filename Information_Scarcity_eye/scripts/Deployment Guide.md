# Deployment Guide

## Overview

Three facial region analysis modes:

- **Face**: Full face with baseline comparison
- **Eye + Eyebrow**: Upper face region with baseline
- **Eye**: Eye region only with baseline

Each mode supports three models: GPT-4o, Gemma, Qwen

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. File Structure

```
face_baseline_gpt.py
face_baseline_Gemma.py
face_baseline_Qwen.py
eye_eyebrow_baseline_gpt.py
eye_eyebrow_baseline_Gemma.py
eye_eyebrow_baseline_Qwen.py
eye_baseline_gpt.py
eye_baseline_Gemma.py
eye_baseline_Qwen.py
```

### 3. Image Naming Convention

**Face Mode:**

```
<id>_<subject>_<condition>_<emotion>.jpg
Example: 101_F_adult_happy.jpg
```

**Eye + Eyebrow Mode:**

```
cropped_<id>_<subject>_<condition>_<emotion>.jpg
Example: cropped_101_F_adult_happy.jpg
```

**Eye Mode:**

```
cropped_<id1>_<id2>_Rafd090_<id3>_<subject>_<condition>_<emotion>_frontal.jpg
Example: cropped_101_202_Rafd090_303_F_adult_happy_frontal.jpg
```

Valid emotions: `angry`, `disgusted`, `fearful`, `happy`, `sad`, `surprised`

## Configuration

### GPT-4o Scripts

#### API Key

```python
api_key = "your-openai-api-key"
```

#### Image Folder

```python
# face_baseline_gpt.py
experience_folder = r"D:\PythonProject\T-LLM\sun-task\face-comb"

# eye_eyebrow_baseline_gpt.py
experience_folder = r"D:\PythonProject\T-LLM\sun-task\eye-comb"

# eye_baseline_gpt.py
experience_folder = r"D:\PythonProject\T-LLM\sun-task\crop2\stitched"
```

#### Output File

```python
# Modify output filename
excel_output_path = "combined_emotion_results31.xlsx"
```

### Gemma Scripts

#### Model Path

```python
model_id = "/app/data/huggingface_models/google--gemma-3-27b-it"
```

#### Image Folder

```python
image_folder = os.path.expanduser("/home/majc_docker/image_block")
```

#### Output Directory

```python
output_dir = os.path.join(script_dir, "results_Gemma_face")
```

#### GPU Configuration

```python
# Automatic GPU allocation
device_map="auto"
torch_dtype=torch.bfloat16
```

### Qwen Scripts

#### GPU Selection

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
```

#### Model Path

```python
model_id = "/data/huggingface_models/Qwen2.5-VL-72B-Instruct"
```

#### Image Folder

```python
image_folder = "/home/majc/image_block"
```

#### Output Directory

```python
output_dir = os.path.join(script_dir, "results_qwen_face")
```

## Execution

### Face Analysis

```bash
python face_baseline_gpt.py
python face_baseline_Gemma.py
python face_baseline_Qwen.py
```

### Eye + Eyebrow Analysis

```bash
python eye_eyebrow_baseline_gpt.py
python eye_eyebrow_baseline_Gemma.py
python eye_eyebrow_baseline_Qwen.py
```

### Eye Analysis

```bash
python eye_baseline_gpt.py
python eye_baseline_Gemma.py
python eye_baseline_Qwen.py
```

## Output Format

### Columns

- `pic`: Image filename
- `emotion`: Ground truth emotion label
- `textbox`: Descriptive adjective response
- `valence`: Emotional valence (1.00-5.00)
- `type`: Emotion type code (1-6)
- `strength`: Emotion intensity (1.00-5.00)
- `correct_emotion`: Ground truth (duplicate)
- `trial_selection`: Model predicted emotion
- `result`: Correct (1) or incorrect (0)

### File Naming

- GPT: `combined_emotion_results{n}.xlsx`
- Gemma: `gemma_analysis_results_{n}.xlsx` + `.txt`
- Qwen: `qwen_analysis_results_{n}.xlsx` + `qwen_analysis_errors_{n}.xlsx`

## Prompts

### Face Mode

- Baseline: Top image (neutral expression)
- Target: Bottom image (emotional expression)
- Description: "Two faces. Top shows baseline (neutral), bottom shows emotion. Reference baseline to analyze bottom face."

### Eye + Eyebrow Mode

- Baseline: Top pair (neutral)
- Target: Bottom pair (emotional)
- Description: "Two eye pairs. Top shows baseline (neutral), bottom shows emotion. Reference baseline to analyze bottom eyes."

### Eye Mode

- Baseline: Top pair (neutral)
- Target: Bottom pair (emotional)
- Description: "Two eye pairs. Top displays baseline (neutral), bottom displays emotional expression. Reference top to analyze bottom."

## Parameters

### Retry Settings

```python
# GPT
max_retries_prompt_type = 5  # general
max_retries_prompt_type = 1000  # emotion only

# Gemma
max_retries = 6

# Qwen
max_retries = 3
```

### Generation Settings

```python
max_new_tokens = 150
top_p = 1
top_k = None
temperature = 1
do_sample = True
```

## Validation Rules

### Emotion

Must be one of: `angry`, `disgusted`, `fearful`, `happy`, `sad`, `surprised`

### Valence & Strength

- Range: 1.00 to 5.00
- Format: Two decimal places
- Examples: 2.75, 3.90, 4.25

### Emotion Type Mapping

```
angry: 1
disgusted: 2
fearful: 3
happy: 4
sad: 5
surprised: 6
```

## System Requirements

### GPT Mode

- Python 3.9+
- 4GB+ RAM
- Internet connection
- Valid OpenAI API key

### Local Deployment (Gemma/Qwen)

- Python 3.9+
- NVIDIA GPU with CUDA
- 24GB+ VRAM per GPU
- 32GB+ RAM (64GB+ preferred)
- 100GB+ storage

## Error Handling

### GPT

- API retry: 10 attempts, exponential backoff
- Parse retry: 3 attempts
- Timeout: 60 seconds
- Fallback: Random emotion if all retries fail

### Gemma

- Overall retry: 6 attempts
- Inference retry: 3 attempts
- CUDA cache clearing

### Qwen

- Request retry: 4 attempts
- Parse retry: 3 attempts
- Error logging to Excel

## Notes

1. Ensure image naming matches expected patterns
2. Monitor GPU memory usage for local models
3. Backup results regularly
4. Different prompts between face/eye regions
5. Baseline comparison critical for accuracy

# Deployment Guide

## Quick Start

### 1) Install Dependencies

```bash
# Navigate to project directory
cd <project-directory>

# Install dependencies
pip install -r requirements.txt
```

### 2) Choose Running Mode

This project supports two running modes:

#### Mode A: Using OpenAI API (GPT-4o)

```bash
# Set OpenAI API Key
export OPENAI_API_KEY="your-openai-api-key"
```

**Features:**

- No local GPU required
- Requires stable internet connection
- Charged per API call

#### Mode B: Local Model Deployment (Gemma / Qwen)

**Prerequisites:**

- Download and configure corresponding model files locally
- Gemma model path example: `/app/data/huggingface_models/google--gemma-3-27b-it`
- Qwen model path example: `/data/huggingface_models/Qwen2.5-VL-72B-Instruct`
- Configure GPU environment (CUDA)

**GPU Configuration:**

```bash
# Qwen model requires specifying available GPUs (configured in script)
# Set at script beginning: os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8,9"

# Gemma model automatically uses available GPU resources
```

**Features:**

- Requires powerful local GPU resources
- No API call fees
- Data stays in local environment

### 3) Prepare Image Data

Place facial expression images to be analyzed in the specified folder. Image files should follow the naming convention:

- Filename format: `image_<number>.jpg`
- Examples: `image_1.jpg`, `image_2.jpg`, `image_100.jpg`

```bash
# Create image folder
mkdir -p data/images

# Copy your image files to the folder
cp /path/to/your/images/*.jpg data/images/
```

### 4) Run Emotion Analysis Evaluation

Select the corresponding script based on different models and prompt versions:

#### GPT-4o Model (API Mode)

```bash
# Use different prompt versions (4 versions total)
python GPT_prompt1.py
python GPT_prompt2.py
python GPT_prompt3.py
python GPT_prompt4.py
```

#### Gemma Model (Local Deployment)

```bash
# Use different prompt versions (4 versions total)
python Gemma_prompt1.py
python Gemma_prompt2.py
python Gemma_prompt3.py
python Gemma_prompt4.py
```

#### Qwen Model (Local Deployment)

```bash
# Use different prompt versions (4 versions total)
python Qwen_prompt1.py
python Qwen_prompt2.py
python Qwen_prompt3.py
python Qwen_prompt4.py
```

**Note:** Different prompt versions mainly differ in prompt wording details. Choose the appropriate version based on experimental requirements.

---

## System Requirements

### API Mode (GPT-4o)

- **Python Version:** 3.9+ (recommended 3.10/3.11)
- **Operating System:** Linux / macOS / Windows
- **Memory:** Recommended 4GB+
- **Network:** Stable internet connection
- **OpenAI API:** Valid API Key with sufficient quota

### Local Deployment Mode (Gemma / Qwen)

- **Python Version:** 3.9+ (recommended 3.10/3.11)
- **Operating System:** Linux (recommended) / macOS
- **GPU:** NVIDIA GPU with CUDA support
  - Qwen model: Recommended multi-GPU configuration (e.g., 5-9 GPUs, depending on model size)
  - Gemma model: Automatically adapts to available GPUs
- **VRAM:** Recommended 24GB+ per GPU (larger models require more)
- **Memory:** Recommended 32GB+ (64GB+ preferred)
- **Storage:** Sufficient disk space for model files (100GB+)
- **CUDA:** Compatible CUDA version
- **PyTorch:** CUDA-enabled version

### Dependencies

Main dependencies include:

**Common Dependencies:**

- `pandas` - Data processing and Excel file operations
- `requests` - HTTP request library (required for API mode)
- `tqdm` - Progress bar display
- `Pillow` - Image I/O and preprocessing
- `openpyxl` - Excel file operation support

**Additional Dependencies for Local Deployment:**

- `torch` - PyTorch deep learning framework
- `transformers` - Hugging Face model library

### Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Use Tsinghua mirror for acceleration (optional)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## Configuration Instructions

### 1. GPT-4o Model Configuration (API Mode)

#### API Key Configuration

In the `main()` function of `GPT_prompt*.py` scripts, find and replace your API key:

```python
def main():
    # Replace with your actual OpenAI API key
    api_key = "your-openai-api-key-here"
    analyzer = EmotionAnalyzer(api_key, num_workers=8)
    ...
```

#### Image Folder Path

Modify the `image_folder` variable in the script to your image folder path:

```python
# Windows path example
image_folder = r"C:\Users\YourName\Desktop\images\image_block"

# Linux/Mac path example
image_folder = "/path/to/your/images/image_block"
```

#### Output Folder Path

Modify the `output_dir` variable in the script to your desired results path:

```python
# Windows path example
output_dir = r"C:\Users\YourName\Desktop\results\gpt"

# Linux/Mac path example
output_dir = "/path/to/your/results/gpt"
os.makedirs(output_dir, exist_ok=True)
```

#### Adjust Concurrency Parameters

Adjust the number of concurrent worker threads based on your network bandwidth and API rate limits:

```python
# num_workers: Number of concurrent processing threads
# Recommended value: 4-8 (adjust based on OpenAI API rate limits)
analyzer = EmotionAnalyzer(api_key, num_workers=8)
```

### 2. Gemma Model Configuration (Local Deployment)

#### Model Path Configuration

In `Gemma_prompt*.py` scripts, modify the model path to your downloaded model's actual location:

```python
# Modify to your actual Gemma model path
model_id = "/app/data/huggingface_models/google--gemma-3-27b-it"
```

#### Image Folder Path

```python
# Modify to your image folder path
image_folder = os.path.expanduser("/home/username/image_block")

# Or use absolute path
image_folder = "/absolute/path/to/image_block"
```

#### Output Path

Output directory is automatically created in the script directory by default, e.g., `results_Gemma_*` folder. To modify:

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "your_custom_output_folder")
os.makedirs(output_dir, exist_ok=True)
```

#### GPU Configuration

Gemma model uses `device_map="auto"` to automatically allocate GPU resources, usually no manual configuration needed. If encountering VRAM shortage, try:

- Enable quantization (8-bit or 4-bit)
- Reduce `max_new_tokens` parameter
- Use single GPU mode

### 3. Qwen Model Configuration (Local Deployment)

#### GPU Device Configuration

At the beginning of `Qwen_prompt*.py` scripts, configure available GPU devices:

```python
# Specify GPU numbers to use (modify based on your hardware configuration)
# Multi-GPU configuration example
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8,9"

# Single GPU configuration example
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

#### Model Path Configuration

```python
# Modify to your actual Qwen model path
model_id = "/data/huggingface_models/Qwen2.5-VL-72B-Instruct"
```

#### Image Folder Path

```python
# Modify to your image folder path
image_folder = "/home/username/image_block"
```

#### Output Path

Output directory is automatically created in the script directory by default, e.g., `results_qwen_*` folder. Can modify in `main()` function:

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "your_custom_output_folder")
os.makedirs(output_dir, exist_ok=True)
```

### 4. General Parameter Adjustments

#### Modify Number of Repeated Runs

All scripts run multiple times by default to collect multiple evaluation data. Modify the loop count in the `main()` function as needed:

```python
# GPT script example: default 40 times
for i in range(1, 41):  # Modify 41 to your desired count + 1
    ...

# Gemma/Qwen script example: default 41 times
for run_idx in range(1, 42):  # Modify 42 to your desired count + 1
    ...
```

#### Adjust Generation Parameters (Local Deployment Mode)

In local deployment scripts, you can adjust model generation parameters:

```python
generation = self.model.generate(
    **inputs,
    max_new_tokens=150,      # Maximum number of tokens to generate
    top_p=1,                 # Nucleus sampling parameter
    top_k=None,              # Top-k sampling parameter
    temperature=1,           # Temperature parameter (lower is more deterministic)
    do_sample=True,          # Whether to use sampling
)
```

---

## Output Results

### Output File Format

The program generates Excel files, with each run saved as an independent file:

**GPT Model:**

- Filename format: `combined_results01.xlsx`, `combined_results02.xlsx`, ...
- Location: In the configured `output_dir` folder

**Gemma Model:**

- Filename format: `gemma_analysis_results*01.xlsx`, `gemma_analysis_results*02.xlsx`, ...
- Location: In the `results_Gemma_*` folder under the script directory
- Also generates text files (.txt) containing complete response content

**Qwen Model:**

- Filename format: `qwen_analysis_results_01.xlsx`, `qwen_analysis_results_02.xlsx`, ...
- Location: In the `results_qwen_*` folder under the script directory
- Error reports: `qwen_analysis_errors_*.xlsx` (records images that failed processing)

---

## Retry Mechanism and Error Handling

### API Mode (GPT)

- **API Request Retry:** Up to 10 times, exponential backoff strategy
- **Server Errors:** Automatic retry (500, 502, 503, 504)
- **Rate Limiting:** Request overload (429) automatically waits and retries
- **Parsing Failure:** Up to 3 retries
- **Timeout Setting:** Default 120 seconds

### Local Deployment Mode (Gemma / Qwen)

- **Qwen Model:**
  
  - API request retry: Up to 4 times
  - Parsing failure retry: Up to 3 times
  - Automatically records errors to Excel file

- **Gemma Model:**
  
  - Overall processing retry: Up to 6 times
  - Model inference retry: Up to 3 times
  - Detailed error information output to console

- **Memory Management:** Automatic CUDA cache release (`torch.cuda.empty_cache()`)

- **Exception Capture:** Complete traceback output for debugging

---

## Notes

1. 
2. **Path Format:** Windows uses `r"C:\path"` or `"C:\\path"`, Linux/Mac uses `"/path"`
3. **GPU VRAM:** Large models require sufficient VRAM, recommend monitoring GPU usage (`nvidia-smi`)
4. **Concurrency Count:** API mode should consider rate limits, local mode recommends single-thread to avoid GPU contention
5. **Data Backup:** Regularly backup generated result files to avoid overwriting or loss
6. **Prompt Versions:** Different prompt versions will affect results, recommend recording the version used

---

## License

This project follows the MIT License. See LICENSE file for details.

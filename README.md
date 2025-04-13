# LoraCraft: Advanced LLM Fine-Tuning Framework

![LoraCraft Logo](assets/loracraft-logo.png)

LoraCraft is a modern, powerful framework for efficiently fine-tuning Large Language Models (LLMs) using the Hugging Face Transformers Reinforcement Learning (TRL) library. It provides a streamlined approach to fine-tune open-source LLMs like Llama, Mistral, and others with state-of-the-art techniques including QLoRA, template customization, and multi-GPU training.

## Documentation

- [Configuration Options Guide](./docs/configuration-options-guide.md) - Comprehensive overview of all configuration options
- [Platform-Specific Installation Guide](./docs/platform-installation-guide.md) - Detailed setup instructions for Windows, macOS, and Linux

## Features

- **QLoRA/LoRA Fine-Tuning**: Efficiently fine-tune models with 4-bit quantization and Low-Rank Adaptation
- **Flexible Configuration**: Simple JSON config files for easy experimentation and reproducibility
- **Memory Efficient**: Optimized for training on consumer GPUs (even with limited VRAM)
- **Versatile Data Processing**: Support for both structured JSON datasets and raw text files for training
- **Adaptable Templates**: Customize data formatting templates for different training scenarios
- **TensorBoard Integration**: Track your training metrics in real-time
- **Checkpoint Management**: Save and resume training from checkpoints
- **Interactive Model Preview**: Sample outputs during training to monitor quality
- **Multi-GPU Support**: Scale your training across multiple GPUs with DeepSpeed integration
- **Model Evaluation**: Test model quality at checkpoints with custom question sets
- **ExLlama2 Support**: Leverage ExLlama2 for faster inference and training
- **TRL Integration**: Built on the powerful TRL framework from Hugging Face

## Installation

### Prerequisites

- Python 3.11 
- CUDA-compatible GPU (recommended)
- Git

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/hypersniper05/loracraft.git
   cd loracraft
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Platform-Specific Notes

For detailed platform-specific installation instructions, see our [Platform-Specific Installation Guide](./docs/platform-installation-guide.md).

#### Windows

- ExLlama2 integration requires Microsoft Visual C++ Build Tools. If you encounter build issues:
  - Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Or use pre-built wheels: `pip install --find-links=https://github.com/turboderp/exllamav2/releases exllamav2`
- If you encounter CUDA issues, ensure you have the matching CUDA toolkit version for your PyTorch installation

#### Mac

- While basic training works on Apple Silicon Macs, some components like ExLlama2 are not natively supported on macOS
- For Mac users, use the following installation command for optimal compatibility:
  ```bash
  CMAKE_ARGS="-DGGML_METAL=ON -DCMAKE_OSX_ARCHITECTURES=arm64" pip install -r requirements.txt
  ```

#### Linux

- ExLlama2 integration works best on Linux
- Ensure you have the CUDA development tools installed for your distribution

## Quick Start

1. Prepare your configuration file in the `configs` directory (see example configurations)
2. Run the training script:
   ```bash
   python train.py
   ```
3. Follow the interactive prompts to select your configuration and model
4. Monitor training progress through the console and TensorBoard

## Directory Structure

```
├── configs/                  # Configuration files
│   ├── llama3_finetune.json
│   ├── mistral_finetune.json
│   ├── ...
├── docs/                     # Documentation files
│   ├── configuration-options-guide.md
│   ├── platform-installation-guide.md
├── text-generation-webui/    # Web UI for testing models
│   ├── models/               # Base models directory
│   ├── loras/                # Output LoRA adapters directory
│   ├── training/             # Training data and formats
│       ├── datasets/         # Training datasets
│       ├── formats/          # Data formatting templates
├── logs/                     # Training logs
├── questions/                # Evaluation prompt templates
├── accelerate_configs/       # Accelerate configurations for distributed training
│   ├── deepspeed_zero1.yaml  
│   ├── deepspeed_zero2.yaml
│   ├── deepspeed_zero3.yaml
├── train.py                  # Main training script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## Configuration

LoraCraft uses JSON configuration files to define training parameters. Example configuration:

```json
{
  "models_directory": "text-generation-webui\\models",
  "dataset_directory": "text-generation-webui\\training\\datasets",
  "lora_directory": "text-generation-webui\\loras",
  "data_format": "alpaca-format",
  "dataset": "your_dataset",
  "batch_size": 4,
  "gradient_accumulation_steps": 4,
  "epochs": 3,
  "learning_rate": 0.0002,
  "lora_rank": 8,
  "lora_alpha": 16,
  "target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj"
  ],
  "cutoff_len": 512,
  "use_prompts": true,
  "prompts_directory": ".\\questions",
  "prompt_file": "evaluation_questions.json"
}
```

See the [Configuration Options Guide](./docs/configuration-options-guide.md) for detailed explanation of all available options.

## Training Data

LoraCraft supports two primary data formats:

1. **JSON datasets**: Structured data with instruction/response pairs
   ```json
   [
     {
       "instruction": "What is the capital of France?",
       "output": "The capital of France is Paris."
     }
   ]
   ```

2. **Raw text files**: Plain text for continuous pre-training
   - Single file mode: Set `raw_text_file` to the filename (without extension) to process just one file
   - Directory mode: Set `raw_text_file` to a directory name to process all .txt files in that directory

For raw text processing, note these important details:
   - When specifying a single file (e.g., "file1"), only that exact file (file1.txt) will be processed
   - When specifying a directory, all .txt files in that directory will be processed in natural sort order
   - Configure `hard_cut_string`, `overlap_len`, and `newline_favor_len` to control text chunking
   - Set `min_chars` to skip chunks that are too short

The format of structured data is controlled by templates in the `format_directory`.

## Multi-GPU Training

LoraCraft supports distributed training across multiple GPUs using DeepSpeed. To enable:

1. Make sure your configuration includes GPU memory allocation:
   ```json
   {
     "max_memory": {
       "0": "16GiB",
       "1": "16GiB"
     },
     "device_map": "auto"
   }
   ```

2. Run training with the accelerate launcher:
   ```bash
   accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml train.py
   ```

3. Choose the appropriate DeepSpeed configuration based on your needs:
   - `deepspeed_zero1.yaml`: Basic optimization, good for most cases
   - `deepspeed_zero2.yaml`: Better memory efficiency
   - `deepspeed_zero3.yaml`: Maximum memory efficiency, but slower

## Model Evaluation During Training

LoraCraft can evaluate the model's performance during training using a set of predefined questions:

1. Set `use_prompts` to `true` in your configuration
2. Specify `prompts_directory` and `prompt_file`
3. Create a JSON file in the prompts directory with your evaluation questions:
   ```json
   [
     {
       "instruction": "Explain the concept of quantum computing."
     },
     {
       "instruction": "Write a short poem about artificial intelligence."
     }
   ]
   ```
4. During training, the model will generate responses to these questions at regular intervals (controlled by `logging_steps`)
5. Responses are saved to `{lora_directory}/{model}/{model}_outputs.json` for review

## Customizing Templates

LoraCraft supports customizable data formats through template files:

1. Create a template JSON file in the `format_directory`
2. Define the format for your data with placeholders:
   ```json
   {
     "instruction,output": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n%instruction%\n\n### Response:\n%output%"
   }
   ```
3. Specify the template in your configuration with `data_format`

## ExLlama2 Integration

LoraCraft supports ExLlama2 for faster inference and potentially more efficient training:

1. Set `use_exllama` to `true` in your configuration
2. Ensure you have the ExLlama2 library installed:
   ```bash
   pip install exllamav2
   ```
3. Windows users may need to install Visual Studio Build Tools or use pre-built wheels

## Advanced Usage

### Resuming from Checkpoints

The trainer automatically saves checkpoints during training. To resume:
1. Run the training script
2. Select the same model and configuration
3. Choose the checkpoint you want to resume from

### NEFTune Noise Augmentation

Enable [NEFTune](https://arxiv.org/abs/2310.05914) for improved performance:

```json
{
  "neftune_noise_alpha": 5
}
```

### Flash Attention

Enable Flash Attention for faster training (requires GPU support):

```json
{
  "use_attn_implementation": true
}
```

## Roadmap and TODO List

The following features are planned for future releases:

- **ROC and Metal (Mac) Training**
  - Improve support for AMD GPUs with ROCm
  - Enhance Metal acceleration for Apple Silicon Macs
  - Optimize memory usage for Mac training

- **Reinforcement Learning**
  - Add support for RLHF (Reinforcement Learning from Human Feedback)
  - Implement alignment tuning after initial fine-tuning
  - Provide tools for reward model training

- **GGUF Converter**
  - Create tools to merge LoRA adapters with base models
  - Implement GGUF format conversion for llama.cpp compatibility
  - Support quantization options during conversion

- **ExLlama Converter**
  - Develop utilities to join adapters with base models
  - Support conversion to ExLlama2 format
  - Maintain performance optimizations in converted models

- **Multi-LoRA Training**
  - Support for loading multiple adapters before training
  - Implement multi-QLora training capabilities
  - Enable adapter stacking and merging

- **Interactive UI**
  - Develop real-time training progress visualization
  - Create interactive parameter tuning interface
  - Implement live model output previewing
  - Add training metrics dashboards

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:
- Reduce batch size
- Increase gradient accumulation steps
- Reduce sequence length
- Use a smaller model
- Enable gradient checkpointing
- Use DeepSpeed ZeRO-3 configuration

### Training Is Too Slow

To improve training speed:
- Enable Flash Attention (`use_attn_implementation: true`)
- Use DeepSpeed ZeRO-1 or ZeRO-2 instead of ZeRO-3
- Reduce logging frequency
- Optimize data loading with more workers

### ExLlama2 Build Issues

- Windows: Install Visual C++ Build Tools or use pre-built wheels
- Linux: Ensure CUDA development tools are installed
- Mac: ExLlama2 is not fully supported on Apple Silicon

For detailed troubleshooting guidance, see our [Platform-Specific Installation Guide](./docs/platform-installation-guide.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the Transformers and TRL libraries
- The open-source LLM community for their invaluable resources
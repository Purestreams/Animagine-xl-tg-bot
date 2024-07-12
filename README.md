# Animagine-xl-tg-bot
Telegram Bot For Animagine-xl

- [huggingface.co/cagliostrolab/animagine-xl-3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1)

<table class="custom-table">
  <tr>
    <td>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/yq_5AWegnLsGyCYyqJ-1G.png" alt="sample1">
      </div>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/sp6w1elvXVTbckkU74v3o.png" alt="sample4">
      </div>
    </td>
    <td>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/OYBuX1XzffN7Pxi4c75JV.png" alt="sample2">
      </div>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/ytT3Oaf-atbqrnPIqz_dq.png" alt="sample3">
    </td>
    <td>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/0oRq204okFxRGECmrIK6d.png" alt="sample1">
      </div>
      <div class="custom-image-container">
        <img class="custom-image" src="https://cdn-uploads.huggingface.co/production/uploads/6365c8dbf31ef76df4042821/DW51m0HlDuAlXwu8H8bIS.png" alt="sample4">
      </div>
    </td>
  </tr>
</table>
   

Animagine-v2 is a Python-based application that utilizes the Telegram Bot API and the Diffusers library to generate images using a pre-trained diffusion model. The application leverages GPU acceleration with PyTorch and provides efficient memory management techniques.

## Features

- Integration with Telegram Bot API
- task queue for image generating
- Image generation using a pre-trained diffusion model
- GPU acceleration with PyTorch
- Efficient memory management with attention slicing, VAE slicing, and tiling

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Telegram Bot API token
- Required Python packages
- Nvidia GPU (at least 8G memory size recommended)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/animagine-v2.git
    cd animagine-v2
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
   ```sh
   pip install diffusers transformers accelerate safetensors python-telegram-bot --upgrade
   ```

5. Set up your Telegram Bot API token:
    - File in your TOKEN in script
      ```env
      TELEGRAM_TOKEN=your_telegram_bot_token
      ```

## Usage

1. Run the application:
    ```sh
    python animagine-v2.py
    ```
    You may use screen to keep script runing in the background, or other methods to run the script by daemon.

2. Interact with the bot on Telegram to generate images.

## Code Overview

- **animagine-v2.py**: Main application file that initializes the diffusion pipeline, sets up logging, and manages the task queue.
- **requirements.txt**: Lists the required Python packages.

## Memory Management Techniques

- **Attention Slicing**: Reduces memory usage by splitting the attention computation into smaller chunks.
- **VAE Slicing**: Splits the VAE computation into smaller chunks to save memory.
- **VAE Tiling**: Tiles the VAE computation to further optimize memory usage.
- **Xformers Memory Efficient Attention**: Uses memory-efficient attention mechanisms provided by Xformers.

## License

This project is licensed under the GNU License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## GPU with limited memory size

see
- [https://huggingface.co/docs/diffusers/optimization/memory](https://huggingface.co/docs/diffusers/optimization/memory)

## Acknowledgements

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [PyTorch](https://pytorch.org/)
- [Python Telegram Bot](https://github.com/python-telegram-bot/python-telegram-bot)
---

Feel free to customize this README file according to your project's specific details and requirements.

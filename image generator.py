import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to generate an image using Stable Diffusion
def generate_image(prompt, negative_prompt=None, num_inference_steps=50, guidance_scale=7.5, seed=None):
    """
    Generate an image using Stable Diffusion with advanced prompt engineering.

    Args:
        prompt (str): The text prompt describing the desired image.
        negative_prompt (str, optional): Text describing what to avoid in the image.
        num_inference_steps (int, optional): Number of denoising steps. Default is 50.
        guidance_scale (float, optional): How closely to follow the prompt. Default is 7.5.
        seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
        PIL.Image: The generated image.
    """
    # Load the Stable Diffusion model
    model_id = "stabilityai/stable-diffusion-2-1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        logger.info("Loading the Stable Diffusion model...")
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        pipe = pipe.to(device)

        # Enable model offloading if running on GPU with limited memory
        if device == "cuda":
            pipe.enable_model_cpu_offload()

        # Set a random seed for reproducibility (if provided)
        if seed is not None:
            torch.manual_seed(seed)

        logger.info("Generating image...")
        with torch.autocast(device_type=device, dtype=torch_dtype):
            image = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]

        return image

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("CUDA out of memory. Try reducing the number of inference steps or using a smaller model.")
        else:
            logger.error(f"Runtime error: {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None

# Main function
if __name__ == "__main__":
    # Define your prompt and negative prompt
    prompt = "A futuristic man at sunset with flying cars above his resort, highly detailed, 4k resolution, cinematic lighting"
    negative_prompt = "blurry, low quality, distorted, oversaturated, cartoonish"

    # Advanced parameters
    num_inference_steps = 50  # More steps = better quality but slower
    guidance_scale = 7.5      # Higher values = more adherence to the prompt
    seed = 42                 # Optional: Set a seed for reproducibility

    # Generate the image
    image = generate_image(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed
    )

    if image is not None:
        # Save the image
        output_path = "generated_image.png"
        image.save(output_path)
        logger.info(f"Image saved as {output_path}")

        # Optionally, display the image
        image.show()
    else:
        logger.error("Failed to generate image.")
import gradio as gr
import numpy as np
import random
import torch
import spaces

from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from optimization import optimize_pipeline_
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

import math
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from PIL import Image
import os


# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", 
                                                transformer= QwenImageTransformer2DModel.from_pretrained("linoyts/Qwen-Image-Edit-Rapid-AIO", 
                                                                                                         subfolder='transformer',
                                                                                                         torch_dtype=dtype,
                                                                                                         device_map='cuda'),torch_dtype=dtype).to(device)

pipe.load_lora_weights("eigen-ai-labs/eigen-banana-qwen-image-edit", 
                       weight_name="eigen-banana-qwen-image-edit-fp16-lora.safetensors",
                       adapter_name="eigen-banana")
pipe.set_adapters(["eigen-banana"], adapter_weights=[1.])
pipe.fuse_lora(adapter_names=["eigen-banana"], lora_scale=1.0)
pipe.unload_lora_weights()

pipe.transformer.__class__ = QwenImageTransformer2DModel
pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

optimize_pipeline_(pipe, image=[Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))], prompt="prompt")

MAX_SEED = np.iinfo(np.int32).max

@spaces.GPU
def convert_to_anime(
    image,
    prompt,
    seed,
    randomize_seed,
    true_guidance_scale,
    num_inference_steps,
    height,
    width,
    progress=gr.Progress(track_tqdm=True)
):
    if not prompt or prompt.strip() == "":
        prompt = "edit"
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    pil_images = []
    if image is not None:
        if isinstance(image, Image.Image):
            pil_images.append(image.convert("RGB"))
        elif hasattr(image, "name"):
            pil_images.append(Image.open(image.name).convert("RGB"))

    if len(pil_images) == 0:
        raise gr.Error("Please upload an image first.")

    result = pipe(
        image=pil_images,
        prompt=prompt,
        height=height if height != 0 else None,
        width=width if width != 0 else None,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=1,
    ).images[0]

    return result, seed


# --- UI ---
css = '''
#col-container { 
    max-width: 900px; 
    margin: 0 auto; 
    padding: 2rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.gradio-container.light {
    background: linear-gradient(to bottom, #f5f5f7, #ffffff);
}
.gradio-container.dark {
    background: linear-gradient(to bottom, #1a1a1a, #0d0d0d);
}
#title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.light #title {
    color: #1d1d1f;
}
.dark #title {
    color: #f5f5f7;
}
#description {
    text-align: center;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.light #description {
    color: #6e6e73;
}
.dark #description {
    color: #a1a1a6;
}
.light #description a {
    color: #0071e3;
}
.dark #description a {
    color: #2997ff;
}
.image-container {
    border-radius: 18px;
    overflow: hidden;
}
.light .image-container {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
}
.dark .image-container {
    box-shadow: 0 4px 6px rgba(255, 255, 255, 0.1);
}
#convert-btn {
    background: linear-gradient(180deg, #0071e3 0%, #0077ed 100%);
    border: none;
    border-radius: 12px;
    color: white;
    font-size: 1.1rem;
    font-weight: 500;
    padding: 0.75rem 2rem;
    transition: all 0.3s ease;
}
#convert-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(0, 113, 227, 0.3);
}
'''

def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024
    
    original_width, original_height = image.size
    
    if original_width > original_height:
        new_width = 1024
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = 1024
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)
        
    # Ensure dimensions are multiples of 8
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    return new_width, new_height


with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("# 🍌 Eigen-Banana-Qwen-Image-Edit: Fast Image Editing with Qwen-Image-Edit LoRA", elem_id="title")
        gr.Markdown(
            """
            Fast image editing powered by Qwen-Image-Edit with Eigen-Banana LoRA ✨
            <br>
            <div style='text-align: center; margin-top: 1rem;'>
                <a href='https://huggingface.co/spaces/akhaliq/anycoder' target='_blank' style='color: #0071e3; text-decoration: none; font-weight: 500;'>Built with anycoder</a>
            </div>
            """,
            elem_id="description"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(
                    label="Upload Photo", 
                    type="pil",
                    elem_classes="image-container"
                )
                
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your editing instruction (e.g., 'Convert this photo to anime style')",
                    lines=2,
                    value="Edit"
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                    true_guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                    num_inference_steps = gr.Slider(label="Inference Steps", minimum=1, maximum=40, step=1, value=4)
                    height = gr.Slider(label="Height", minimum=256, maximum=2048, step=8, value=1024, visible=False)
                    width = gr.Slider(label="Width", minimum=256, maximum=2048, step=8, value=1024, visible=False)
                
                convert_btn = gr.Button("Edit", variant="primary", elem_id="convert-btn", size="lg")

            with gr.Column(scale=1):
                result = gr.Image(
                    label="Result", 
                    interactive=False,
                    elem_classes="image-container"
                )
    
    inputs = [
        image, prompt, seed, randomize_seed, true_guidance_scale, 
        num_inference_steps, height, width
    ]
    outputs = [result, seed]

    # Convert button click
    convert_btn.click(
        fn=convert_to_anime, 
        inputs=inputs, 
        outputs=outputs
    )

    # Image upload triggers dimension update
    image.upload(
        fn=update_dimensions_on_upload,
        inputs=[image],
        outputs=[width, height]
    )

# Setup ngrok for public URL access
from pyngrok import ngrok, conf

# Set ngrok auth token
ngrok.set_auth_token("36HN7sKBiZlNScQ2ixdJORiZd3r_5kTpS6Wu1RL76xy7MPxCz")

# Launch Gradio app on local port
demo.launch(share=False, server_port=7860)

# Create ngrok tunnel
public_url = ngrok.connect(7860)
print(f"\n🌐 Public URL: {public_url}\n")
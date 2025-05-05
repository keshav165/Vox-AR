import torch
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh
import os
import imageio
import config

def generate_3d_model(prompt: str, output_dir: str = None, guidance_scale: float = None):
    output_dir = output_dir or config.OUTPUT_DIR
    guidance_scale = guidance_scale or config.GUIDANCE_SCALE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    print("Selected device:", device)
    try:
        xm = load_model('transmitter', device=device)
        model = load_model('text300M', device=device)
        diffusion = diffusion_from_config(load_config('diffusion'))
        os.makedirs(output_dir, exist_ok=True)
        latents = sample_latents(
            batch_size=1,
            model=model,
            diffusion=diffusion,
            guidance_scale=guidance_scale,
            model_kwargs=dict(texts=[prompt]),
            progress=True,
            clip_denoised=True,
            use_fp16=True,
            use_karras=True,
            karras_steps=64,
            sigma_min=1e-3,
            sigma_max=160,
            s_churn=0,
        )
        render_mode = getattr(config, 'RENDER_MODE', 'nerf')
        size = 64
        cameras = create_pan_cameras(size, device)
        for i, latent in enumerate(latents):
            images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
            # Determine next model number
            existing = [f for f in os.listdir(output_dir) if f.startswith('model_') and (f.endswith('.obj') or f.endswith('.ply') or f.endswith('.gif'))]
            nums = [int(f.split('_')[1].split('.')[0]) for f in existing if '_' in f and f.split('_')[1].split('.')[0].isdigit()]
            model_num = max(nums) + 1 if nums else 1
            gif_path = os.path.join(output_dir, f'model_{model_num}.gif')
            imageio.mimsave(gif_path, images, duration=0.1, loop=0)
            t = decode_latent_mesh(xm, latent).tri_mesh()
            ply_path = os.path.join(output_dir, f'model_{model_num}.ply')
            obj_path = os.path.join(output_dir, f'model_{model_num}.obj')
            with open(ply_path, 'wb') as f:
                t.write_ply(f)
            with open(obj_path, 'w') as f:
                t.write_obj(f)
        return gif_path, ply_path, obj_path
    except Exception as e:
        print(f"Error generating 3D model: {e}")
        raise

if __name__ == "__main__":
    prompt = "a shark"
    output_dir = "generated_models"
    
    print(f"Generating 3D model for prompt: '{prompt}'")
    gif_path, ply_path, obj_path = generate_3d_model(prompt, output_dir)
    print(f"\nGeneration complete!")
    print(f"GIF saved to: {gif_path}")
    print(f"PLY mesh saved to: {ply_path}")
    print(f"OBJ mesh saved to: {obj_path}")

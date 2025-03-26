import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
from diffusers.image_processor import VaeImageProcessor
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from starlette.responses import StreamingResponse
from huggingface_hub import snapshot_download
from utils import resize_and_crop, resize_and_padding, init_weight_dtype  # Match Gradio imports

app = FastAPI(title="CatVTON Virtual Try-On API")

# Configuration (match Gradio exactly)
WIDTH, HEIGHT = 768, 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION = "bf16"  # Match Gradio default
USE_TF32 = True  # Match Gradio default

# Download model weights (same as Gradio)
repo_path = snapshot_download(repo_id="zhengchong/CatVTON")

# Initialize CatVTON Pipeline (exact match to Gradio)
pipeline = CatVTONPipeline(
    base_ckpt="booksforcharlie/stable-diffusion-inpainting",
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(MIXED_PRECISION),  # Use Gradio's utility
    use_tf32=USE_TF32,
    device=DEVICE
)

# Initialize Image Processors
vae_processor = VaeImageProcessor(vae_scale_factor=8)
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

# Initialize AutoMasker
automasker = AutoMasker(
    densepose_ckpt=f"{repo_path}/DensePose",
    schp_ckpt=f"{repo_path}/SCHP",
    device=DEVICE
)

def preprocess_image(img: Image.Image, is_cloth: bool = False) -> torch.Tensor:
    if is_cloth:
        img = resize_and_padding(img, (WIDTH, HEIGHT))
    else:
        img = resize_and_crop(img, (WIDTH, HEIGHT))
    return vae_processor.preprocess(img, HEIGHT, WIDTH)[0]

def preprocess_mask(mask: Image.Image) -> torch.Tensor:
    mask = resize_and_crop(mask, (WIDTH, HEIGHT))
    return mask_processor.preprocess(mask, HEIGHT, WIDTH)[0]

def image_to_bytes(img: Image.Image) -> BytesIO:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    return buffered

@app.post("/predict")
async def predict(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    cloth_type: str = "upper"
):
    try:
        person_data = await person_image.read()
        cloth_data = await cloth_image.read()
        person_img = Image.open(BytesIO(person_data)).convert("RGB")
        cloth_img = Image.open(BytesIO(cloth_data)).convert("RGB")

        person_img_processed = resize_and_crop(person_img, (WIDTH, HEIGHT))
        cloth_img_processed = resize_and_padding(cloth_img, (WIDTH, HEIGHT))

        mask_result = automasker(person_img_processed, cloth_type)
        mask_img = mask_result["mask"]
        if mask_img is None:
            raise HTTPException(status_code=500, detail="Failed to generate mask")
        mask_img = mask_processor.blur(mask_img, blur_factor=9)

        person_tensor = preprocess_image(person_img_processed).unsqueeze(0).to(DEVICE)
        cloth_tensor = preprocess_image(cloth_img_processed, is_cloth=True).unsqueeze(0).to(DEVICE)
        mask_tensor = preprocess_mask(mask_img).unsqueeze(0).to(DEVICE)

        generator = torch.Generator(device=DEVICE).manual_seed(42)
        with torch.no_grad():
            result = pipeline(
                image=person_tensor,
                condition_image=cloth_tensor,
                mask=mask_tensor,
                num_inference_steps=50,
                guidance_scale=2.5,
                generator=generator,
                height=HEIGHT,
                width=WIDTH
            )[0]

        result_bytes = image_to_bytes(result)
        return StreamingResponse(result_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
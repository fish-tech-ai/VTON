import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from io import BytesIO
from diffusers.image_processor import VaeImageProcessor
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import resize_and_crop, resize_and_padding, init_weight_dtype
from google.cloud import storage
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import shutil

app = FastAPI(title="CatVTON Virtual Try-On API")

# Configuration
WIDTH, HEIGHT = 768, 1024
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIXED_PRECISION = "bf16"
USE_TF32 = True
MODELS_BUCKET_NAME = "style-me-models"  # Your GCS bucket name
IMAGES_BUCKET_NAME = "style-me-image"  # Your GCS bucket name
AES_KEY = bytes.fromhex(os.getenv("AES_256").strip())  # Replace with your 32-byte AES key
LOCAL_MODEL_DIR = "/tmp/models"  # Temporary directory for model storage

# Google Cloud Storage client
storage_client = storage.Client()

# Global variables for pipeline and masker (to be initialized later)
pipeline = None
automasker = None
vae_processor = None
mask_processor = None


def download_from_gcs(bucket_name, source_path, local_path):
    bucket = storage_client.bucket(bucket_name)
    prefix = source_path + "/"
    blobs = bucket.list_blobs(prefix=prefix)
    os.makedirs(local_path, exist_ok=True)
    for blob in blobs:
        relative_path = blob.name[len(prefix):]
        local_file_path = os.path.join(local_path, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        blob.download_to_filename(local_file_path)


# Startup event to download models and initialize pipeline
@app.on_event("startup")
async def startup_event():
    global pipeline, automasker, vae_processor, mask_processor

    # Download models from GCS
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    download_from_gcs(MODELS_BUCKET_NAME, "stable-diffusion-inpainting",
                      f"{LOCAL_MODEL_DIR}/stable-diffusion-inpainting")
    download_from_gcs(MODELS_BUCKET_NAME, "catvton", f"{LOCAL_MODEL_DIR}/catvton")

    # Initialize CatVTON Pipeline
    pipeline = CatVTONPipeline(
        base_ckpt=f"{LOCAL_MODEL_DIR}/stable-diffusion-inpainting",
        attn_ckpt=f"{LOCAL_MODEL_DIR}/catvton",
        attn_ckpt_version="mix",
        weight_dtype=init_weight_dtype(MIXED_PRECISION),
        use_tf32=USE_TF32,
        device=DEVICE
    )

    # Initialize Image Processors
    vae_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                       do_convert_grayscale=True)

    # Initialize AutoMasker
    automasker = AutoMasker(
        densepose_ckpt=f"{LOCAL_MODEL_DIR}/catvton/DensePose",
        schp_ckpt=f"{LOCAL_MODEL_DIR}/catvton/SCHP",
        device=DEVICE
    )


def fetch_image_from_gcs(image_id: str) -> bytes:
    bucket = storage_client.bucket(IMAGES_BUCKET_NAME)
    blob = bucket.blob(f"{image_id}.WEBP")
    return blob.download_as_bytes()


def decode_image(encrypted_data: bytes) -> bytes:
    nonce = encrypted_data[:16]
    tag = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]
    cipher = Cipher(algorithms.AES(AES_KEY), modes.GCM(nonce), backend=default_backend())
    decryptor = cipher.decryptor()
    try:
        decrypted_data = decryptor.update(ciphertext) + decryptor.finalize_with_tag(tag)
    except ValueError as e:
        raise Exception("Decryption failed: The data is corrupted or the tag is invalid.") from e
    return decrypted_data


def encode_image(image_data: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(AES_KEY), modes.GCM(), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(image_data) + encryptor.finalize()
    encrypted_data = encryptor.nonce + encryptor.tag + ciphertext
    return encrypted_data


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


def run_inference(person_img: Image.Image, cloth_img: Image.Image, cloth_type: str) -> Image.Image:
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
    return result


@app.post("/predict")
async def predict(
        person_image_id: str,
        cloth_upper_image_id: str = None,
        cloth_lower_image_id: str = None,
        cloth_overall_image_id: str = None,
):
    try:
        # Validate clothing ID logic
        cloth_ids = [cloth_upper_image_id, cloth_lower_image_id, cloth_overall_image_id]
        provided_cloth_ids = [cid for cid in cloth_ids if cid is not None]
        if len(provided_cloth_ids) == 0:
            raise HTTPException(status_code=400, detail="At least one cloth ID must be provided")
        if cloth_overall_image_id and (cloth_upper_image_id or cloth_lower_image_id):
            raise HTTPException(status_code=400, detail="Overall cloth ID cannot be combined with upper or lower IDs")

        # Ensure pipeline and automasker are initialized
        if pipeline is None or automasker is None:
            raise HTTPException(status_code=503, detail="Service not ready: Models are still loading")

        # Fetch and decode person image
        person_data_encrypted = fetch_image_from_gcs(person_image_id)
        person_data = decode_image(person_data_encrypted)
        person_img = Image.open(BytesIO(person_data)).convert("RGB")

        # Handle single inference (overall or single cloth)
        if cloth_overall_image_id:
            cloth_data_encrypted = fetch_image_from_gcs(cloth_overall_image_id)
            cloth_data = decode_image(cloth_data_encrypted)
            cloth_img = Image.open(BytesIO(cloth_data)).convert("RGB")
            result = run_inference(person_img, cloth_img, "overall")
        elif cloth_upper_image_id and not cloth_lower_image_id:
            cloth_data_encrypted = fetch_image_from_gcs(cloth_upper_image_id)
            cloth_data = decode_image(cloth_data_encrypted)
            cloth_img = Image.open(BytesIO(cloth_data)).convert("RGB")
            result = run_inference(person_img, cloth_img, "upper")
        elif cloth_lower_image_id and not cloth_upper_image_id:
            cloth_data_encrypted = fetch_image_from_gcs(cloth_lower_image_id)
            cloth_data = decode_image(cloth_data_encrypted)
            cloth_img = Image.open(BytesIO(cloth_data)).convert("RGB")
            result = run_inference(person_img, cloth_img, "lower")
        # Handle two inferences (upper and lower)
        elif cloth_upper_image_id and cloth_lower_image_id:
            # First inference: Upper cloth
            cloth_upper_data_encrypted = fetch_image_from_gcs(cloth_upper_image_id)
            cloth_upper_data = decode_image(cloth_upper_data_encrypted)
            cloth_upper_img = Image.open(BytesIO(cloth_upper_data)).convert("RGB")
            intermediate_result = run_inference(person_img, cloth_upper_img, "upper")
            # Second inference: Lower cloth on intermediate result
            cloth_lower_data_encrypted = fetch_image_from_gcs(cloth_lower_image_id)
            cloth_lower_data = decode_image(cloth_lower_data_encrypted)
            cloth_lower_img = Image.open(BytesIO(cloth_lower_data)).convert("RGB")
            result = run_inference(intermediate_result, cloth_lower_img, "lower")
        else:
            raise HTTPException(status_code=400, detail="Invalid cloth ID combination")

        # Convert result to bytes and encode with AES
        result_bytes = image_to_bytes(result)
        encrypted_result = encode_image(result_bytes.getvalue())
        return encrypted_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
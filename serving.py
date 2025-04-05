import torch
from fastapi import FastAPI, HTTPException, Response
from PIL import Image
from io import BytesIO
from diffusers.image_processor import VaeImageProcessor
from model.pipeline import CatVTONPipeline
from model.cloth_masker import AutoMasker
from vton_utils import resize_and_crop, resize_and_padding, init_weight_dtype
from google.cloud import storage
from Crypto.Cipher import AES
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
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
        logger.info(f"Downloaded {blob.name} to {local_file_path}")


# Startup event to download models and initialize pipeline
@app.on_event("startup")
async def startup_event():
    global pipeline, automasker, vae_processor, mask_processor

    # Define model paths
    sd_path = f"{LOCAL_MODEL_DIR}/stable-diffusion-inpainting"
    catvton_path = f"{LOCAL_MODEL_DIR}/catvton"

    # Check if models already exist
    models_exist = os.path.exists(sd_path) and os.path.exists(catvton_path)

    if not models_exist:
        logger.info("Models not found in /tmp/models, downloading from GCS...")
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
        download_from_gcs(MODELS_BUCKET_NAME, "stable-diffusion-inpainting", sd_path)
        download_from_gcs(MODELS_BUCKET_NAME, "catvton", catvton_path)
    else:
        logger.info("Models already exist in /tmp/models, skipping download.")

    # Initialize CatVTON Pipeline
    try:
        logger.info("Initializing CatVTON Pipeline...")
        pipeline = CatVTONPipeline(
            base_ckpt=sd_path,
            attn_ckpt=catvton_path,
            attn_ckpt_version="mix",
            weight_dtype=init_weight_dtype(MIXED_PRECISION),
            use_tf32=USE_TF32,
            device=DEVICE
        )
    except Exception as e:
        logger.exception(f"Failed to initialize CatVTON Pipeline: {str(e)}")
        raise

    # Initialize Image Processors
    logger.info("Initializing image processors...")
    vae_processor = VaeImageProcessor(vae_scale_factor=8)
    mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True,
                                       do_convert_grayscale=True)

    # Initialize AutoMasker
    try:
        logger.info("Initializing AutoMasker...")
        automasker = AutoMasker(
            densepose_ckpt=f"{catvton_path}/DensePose",
            schp_ckpt=f"{catvton_path}/SCHP",
            device=DEVICE
        )
    except Exception as e:
        logger.exception(f"Failed to initialize AutoMasker: {str(e)}")
        raise

    logger.info("Startup completed successfully.")


def fetch_image_from_gcs(image_id: str) -> bytes:
    logger.info(f"Fetching image from GCS: {image_id}")
    bucket = storage_client.bucket(IMAGES_BUCKET_NAME)
    blob = bucket.blob(f"{image_id}.WEBP")
    data = blob.download_as_bytes()
    logger.info(f"Fetched data length: {len(data)}, first 32 bytes: {data[:32].hex()}")
    return data


def decode_image(encrypted_data: bytes) -> bytes:
    nonce = encrypted_data[:16]
    tag = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]
    cipher = AES.new(AES_KEY, AES.MODE_GCM, nonce=nonce)
    try:
        decrypted_data = cipher.decrypt_and_verify(ciphertext, tag)
    except ValueError as e:
        raise Exception("Decryption failed: The data is corrupted or the tag is invalid.") from e
    return decrypted_data


def encode_image(image_data: bytes) -> bytes:
    cipher = AES.new(AES_KEY, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(image_data)
    encrypted_data = cipher.nonce + tag + ciphertext
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


def run_inference(person_img: Image.Image, cloth_img: Image.Image, cloth_type: str, inference_steps: int) -> Image.Image:
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
            num_inference_steps=inference_steps,
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
        inference_steps: int = 20
):
    try:
        logger.info(
            f"Received request: person_image_id={person_image_id}, cloth_upper_image_id={cloth_upper_image_id}, "
            f"cloth_lower_image_id={cloth_lower_image_id}, cloth_overall_image_id={cloth_overall_image_id}")

        # Validate clothing ID logic
        cloth_ids = [cloth_upper_image_id, cloth_lower_image_id, cloth_overall_image_id]
        provided_cloth_ids = [cid for cid in cloth_ids if cid is not None]

        if len(provided_cloth_ids) == 0:
            raise HTTPException(status_code=400, detail="At least one cloth ID must be provided")

        # If overall ID is provided, upper and lower IDs must not be present
        if cloth_overall_image_id and (cloth_upper_image_id or cloth_lower_image_id):
            raise HTTPException(status_code=400, detail="Overall cloth ID cannot be combined with upper or lower IDs")

        # Ensure pipeline and automasker are initialized
        if pipeline is None or automasker is None:
            logger.error("Pipeline or automasker not initialized")
            raise HTTPException(status_code=503, detail="Service not ready: Models are still loading")

        # Fetch and decode person image
        logger.info(f"Fetching person image: {person_image_id}")
        person_data_encrypted = fetch_image_from_gcs(person_image_id)
        logger.info("Decoding person image")
        person_data = decode_image(person_data_encrypted)
        person_img = Image.open(BytesIO(person_data)).convert("RGB")

        # Handle inference based on provided cloth IDs
        result = person_img  # Start with the original person image

        # Case 1: Only overall cloth ID
        if cloth_overall_image_id:
            logger.info(f"Fetching overall cloth image: {cloth_overall_image_id}")
            cloth_data_encrypted = fetch_image_from_gcs(cloth_overall_image_id)
            logger.info("Decoding overall cloth image")
            cloth_data = decode_image(cloth_data_encrypted)
            cloth_img = Image.open(BytesIO(cloth_data)).convert("RGB")
            logger.info("Running inference for overall cloth")
            result = run_inference(result, cloth_img, "overall", inference_steps)

        # Case 2: Upper and/or lower cloth IDs
        elif cloth_upper_image_id or cloth_lower_image_id:
            if cloth_upper_image_id and cloth_lower_image_id:
                inference_steps = 15
            if cloth_upper_image_id:
                logger.info(f"Fetching upper cloth image: {cloth_upper_image_id}")
                cloth_data_encrypted = fetch_image_from_gcs(cloth_upper_image_id)
                logger.info("Decoding upper cloth image")
                cloth_data = decode_image(cloth_data_encrypted)
                cloth_img = Image.open(BytesIO(cloth_data)).convert("RGB")
                logger.info("Running inference for upper cloth")
                result = run_inference(result, cloth_img, "upper", inference_steps)

            # Second inference: Lower cloth if provided, using result from upper (or original if no upper)
            if cloth_lower_image_id:
                logger.info(f"Fetching lower cloth image: {cloth_lower_image_id}")
                cloth_data_encrypted = fetch_image_from_gcs(cloth_lower_image_id)
                logger.info("Decoding lower cloth image")
                cloth_data = decode_image(cloth_data_encrypted)
                cloth_img = Image.open(BytesIO(cloth_data)).convert("RGB")
                logger.info("Running inference for lower cloth")
                result = run_inference(result, cloth_img, "lower", inference_steps)

        # Convert result to WEBP bytes (unencrypted)
        logger.info("Converting result to WEBP")
        webp_buffer = BytesIO()
        result.save(webp_buffer, format="WEBP")
        webp_bytes = webp_buffer.getvalue()

        # Return as unencrypted WEBP image
        return Response(
            content=webp_bytes,
            media_type="image/webp",
            headers={"Content-Disposition": "attachment; filename=result.webp"}
        )

    except Exception as e:
        logger.exception(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
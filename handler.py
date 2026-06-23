import os
import sys
import math
import base64
import traceback
import random
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image

import runpod

# ==============================================================================
# 1. 基础全局配置
# ==============================================================================
WEIGHTS_DIR = os.environ.get("FITDIT_WEIGHTS_DIR", "/runpod-volume/FitDiT")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = os.environ.get("FITDIT_FP16", "1") == "1"
CPU_OFFLOAD = os.environ.get("FITDIT_CPU_OFFLOAD", "0") == "1"

# 用户友好标签到模型内部品类的映射
CATEGORY_MAP = {
    "upper_body": "Upper-body",
    "upper-body": "Upper-body",
    "tops": "Upper-body",
    "lower_body": "Lower-body",
    "lower-body": "Lower-body",
    "bottoms": "Lower-body",
    "dresses": "Dresses",
    "full_body": "Dresses",
    "one-pieces": "Dresses",
}

# 动态添加 FitDiT 源码路径到环境变量，确保可以正常 import 内部模块
src_dir = os.path.join(WEIGHTS_DIR, "src")
fitdit_root = WEIGHTS_DIR

for p in [fitdit_root, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ==============================================================================
# 2. 图像处理工具函数
# ==============================================================================
# def decode_base64_image(b64_str: str) -> Image.Image:
#     """将前端传来的 Base64 字符串（含或不含 data-URI 前缀）解码为 PIL Image"""
#     if "," in b64_str:
#         b64_str = b64_str.split(",", 1)[1]
#     return Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")

def decode_image(image_str: str) -> Image.Image:
    if image_str.startswith("http://") or image_str.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(image_str) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")
    if "," in image_str:
        image_str = image_str.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(image_str))).convert("RGB")

def encode_image_to_base64(image: Image.Image, fmt: str = "JPEG", quality: int = 95) -> str:
    """将 PIL Image 编码为带 data-URI 前缀的 Base64 字符串"""
    buf = BytesIO()
    image.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255)):
    """保持宽高比缩放图片，并在周围填充纯色边框到目标尺寸"""
    old_width, old_height = im.size
    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)

    im_resized = im.resize(new_size, Image.LANCZOS)
    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    canvas = Image.new("RGB", (new_width, new_height), pad_color)
    canvas.paste(im_resized, (pad_w, pad_h))
    return canvas, pad_w, pad_h


def unpad_and_resize(padded_im, pad_w, pad_h, orig_w, orig_h):
    """去除缩放时添加的边缘填充，并将图片精准恢复到最原始的尺寸"""
    w, h = padded_im.size
    cropped = padded_im.crop((pad_w, pad_h, w - pad_w, h - pad_h))
    return cropped.resize((orig_w, orig_h), Image.LANCZOS)


def resize_for_detection(img, target_size=768):
    """等比例缩放图片，使较短的边等于目标尺寸（用于姿态与语义分割检测）"""
    w, h = img.size
    scale = target_size / min(w, h)
    return img.resize((int(round(w * scale)), int(round(h * scale))), Image.LANCZOS)


# ==============================================================================
# 3. 模型加载逻辑 (冷启动时运行一次)
# ==============================================================================
pipeline = None
dwprocessor = None
parsing_model = None


def load_models():
    global pipeline, dwprocessor, parsing_model

    print(f"[FitDiT] Loading models from {WEIGHTS_DIR} on {DEVICE} ...")

    # 1. 验证核心权重文件是否存在
    required = [
        os.path.join(WEIGHTS_DIR, "transformer_garm"),
        os.path.join(WEIGHTS_DIR, "transformer_vton"),
        os.path.join(WEIGHTS_DIR, "vae"),
        os.path.join(WEIGHTS_DIR, "pose_guider", "diffusion_pytorch_model.bin"),
        os.path.join(WEIGHTS_DIR, "dwpose"),
        os.path.join(WEIGHTS_DIR, "humanparsing"),
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(f" - {p}" for p in missing)
            + "\n\nRun download_weights.py first."
        )

    # 2. 验证 FitDiT 源码是否存在
    for mod in ["gradio_sd3.py", "src/pipeline_stable_diffusion_3_tryon.py"]:
        path = os.path.join(WEIGHTS_DIR, mod)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"FitDiT source not found at {path}. "
                f"Clone the repo into the weights directory:\n"
                f" git clone https://github.com/BoyuanJiang/FitDiT.git {WEIGHTS_DIR}"
            )

    weight_dtype = torch.float16 if USE_FP16 else torch.bfloat16

    # 3. 动态导入模型组件
    from transformers import CLIPVisionModelWithProjection
    from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
    from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
    from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton
    from src.pose_guider import PoseGuider
    from preprocess.dwpose import DWposeDetector
    from preprocess.humanparsing.run_parsing import Parsing

    # 4. 加载试衣网络分支
    transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(
        os.path.join(WEIGHTS_DIR, "transformer_garm"), torch_dtype=weight_dtype
    )
    transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(
        os.path.join(WEIGHTS_DIR, "transformer_vton"), torch_dtype=weight_dtype
    )

    # 5. 初始化并加载姿态引导网络
    pose_guider = PoseGuider(
        conditioning_embedding_channels=1536,
        conditioning_channels=3,
        block_out_channels=(32, 64, 256, 512),
    )
    pose_guider.load_state_dict(
        torch.load(os.path.join(WEIGHTS_DIR, "pose_guider", "diffusion_pytorch_model.bin"), map_location="cpu")
    )
    pose_guider.to(device=DEVICE, dtype=weight_dtype)

    # 6. 加载视觉特征提取模型
    clip_large_path = os.path.join(WEIGHTS_DIR, "clip-vit-large-patch14")
    clip_bigG_path = os.path.join(WEIGHTS_DIR, "clip-vit-bigG-14")

    image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(
        # "openai/clip-vit-large-patch14", torch_dtype=weight_dtype
        clip_large_path, torch_dtype=weight_dtype
    )
    image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
        # "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=weight_dtype
        clip_bigG_path, torch_dtype=weight_dtype
    )
    image_encoder_large.to(DEVICE)
    image_encoder_bigG.to(DEVICE)

    # 7. 组装 Stable Diffusion 3 试衣流水线
    pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
        WEIGHTS_DIR,
        torch_dtype=weight_dtype,
        transformer_garm=transformer_garm,
        transformer_vton=transformer_vton,
        pose_guider=pose_guider,
        image_encoder_large=image_encoder_large,
        image_encoder_bigG=image_encoder_bigG,
    )

    # 处理显存优化策略
    if CPU_OFFLOAD:
        pipeline.enable_model_cpu_offload()
        preprocess_device = "cpu"
    else:
        pipeline.to(DEVICE)
        preprocess_device = DEVICE

    # 8. 初始化前置姿态与分割检测模型
    dwprocessor = DWposeDetector(model_root=WEIGHTS_DIR, device=preprocess_device)
    parsing_model = Parsing(model_root=WEIGHTS_DIR, device=preprocess_device)

    print("[FitDiT] All models loaded successfully.")


# ==============================================================================
# 4. 自动生成掩码与姿态 (当前端未传自定义 mask 时使用)
# ==============================================================================
def generate_mask_and_pose(person_img: Image.Image, category: str):
    """提取人体姿态骨骼并进行语义分割，自动输出试衣遮罩"""
    from src.utils_mask import get_mask_location

    det_img = resize_for_detection(person_img)

    # 估计人体姿态点
    pose_img_bgr, _keypoints, _, candidate = dwprocessor(np.array(det_img)[:, :, ::-1])
    candidate[candidate < 0] = 0
    candidate = candidate[0]
    candidate[:, 0] *= det_img.width
    candidate[:, 1] *= det_img.height

    pose_image = Image.fromarray(pose_img_bgr[:, :, ::-1])  # BGR 转换为 RGB

    # 运行人体分割模型
    model_parse, _ = parsing_model(det_img)

    # 生成局部涂抹掩码 (Mask)
    mask, mask_gray = get_mask_location(
        category, model_parse, candidate,
        model_parse.width, model_parse.height,
        offset_top=0, offset_bottom=0, offset_left=0, offset_right=0,
    )

    # 将生成的图片分辨率还原回原图的大小尺寸
    mask = mask.resize(person_img.size).convert("L")
    mask_gray = mask_gray.resize(person_img.size).convert("L")
    pose_image = pose_image.resize(person_img.size)

    return mask, mask_gray, pose_image


# ==============================================================================
# 5. RunPod 请求监听与任务分发器
# ==============================================================================
def handler(job):
    """处理 Serverless 调用的核心逻辑"""
    try:
        if pipeline is None:
            return {"error": "Model pipeline not loaded. Check container logs."}

        inp = job.get("input", {})

        # 1. 验证必填字段
        model_b64 = inp.get("model_image")
        garment_b64 = inp.get("garment_image")
        if not model_b64 or not garment_b64:
            return {"error": "Both 'model_image' and 'garment_image' are required."}

        # 2. 参数解析与安全拦截
        raw_category = inp.get("category", "upper_body").lower().strip()
        category = CATEGORY_MAP.get(raw_category)
        if category is None:
            return {
                "error": f"Unknown category '{raw_category}'. Use: upper_body, lower_body, dresses, tops, bottoms, one-pieces."}

        steps = int(inp.get("steps", 40))
        guidance_scale = float(inp.get("guidance_scale", 2.0))
        seed = int(inp.get("seed", -1))
        num_images = min(max(int(inp.get("num_images", 1)), 1), 4)
        resolution = inp.get("resolution", "1152x1536")

        if resolution not in ["768x1024", "1152x1536", "1536x2048"]:
            return {"error": f"Invalid resolution '{resolution}'. Use: 768x1024, 1152x1536, 1536x2048."}

        new_width, new_height = map(int, resolution.split("x"))

        # 3. 解码输入的 Base64 图像
        person_img = decode_image(model_b64)
        garment_img = decode_image(garment_b64)
        original_size = person_img.size  # 保存用户的原始尺寸 (W, H)

        # 4. 获取控制遮罩与姿态图
        mask_b64 = inp.get("mask_image")
        if mask_b64:
            # 如果用户自己上传了高级遮罩，直接解码使用
            mask = decode_image(mask_b64).convert("L")
            det_img = resize_for_detection(person_img)
            pose_img_bgr, _, _, _ = dwprocessor(np.array(det_img)[:, :, ::-1])
            pose_image = Image.fromarray(pose_img_bgr[:, :, ::-1]).resize(person_img.size)
        else:
            # 否则，智能全自动分析生成
            mask, _mask_gray, pose_image = generate_mask_and_pose(person_img, category)

        # 5. 标准化缩放与填充 (对齐到模型可识别的底稿)
        person_resized, pad_w, pad_h = pad_and_resize(person_img, new_width, new_height)
        garment_resized, _, _ = pad_and_resize(garment_img, new_width, new_height)
        mask_resized, _, _ = pad_and_resize(mask, new_width, new_height, pad_color=(0, 0, 0))
        mask_resized = mask_resized.convert("L")
        pose_resized, _, _ = pad_and_resize(pose_image, new_width, new_height, pad_color=(0, 0, 0))

        # 6. 处理随机种子
        if seed == -1:
            seed = random.randint(0, 2147483647)

        # 7. 开启闭环推理 (不计算梯度，省显存加速)
        with torch.inference_mode():
            results = pipeline(
                height=new_height,
                width=new_width,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                cloth_image=garment_resized,
                model_image=person_resized,
                mask=mask_resized,
                pose_image=pose_resized,
                num_images_per_prompt=num_images,
            ).images

        # 8. 图像后处理还原，打包回传
        output_images = []
        for img in results:
            img = unpad_and_resize(img, pad_w, pad_h, original_size[0], original_size[1])
            output_images.append(encode_image_to_base64(img))

        return {
            "status": "success",
            "images": output_images,
            "seed": seed,
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


# ==============================================================================
# 6. 容器守护进程入口
# ==============================================================================
if __name__ == "__main__":
    try:
        load_models()
    except Exception:
        traceback.print_exc()
        print("[FitDiT] FATAL: Failed to load models. Container will accept jobs but return errors.")

    # 启动 RunPod Serverless 服务循环
    runpod.serverless.start({"handler": handler})
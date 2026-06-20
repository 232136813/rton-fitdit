import os
import sys
import torch
import runpod
import base64
from PIL import Image
from io import BytesIO

# ----------------------------------------------------
# 1. 精准路径定位：模型和代码都在网盘的同一目录下
# ----------------------------------------------------
model_dir = "/runpod-volume/FitDiT"

# 核心：必须把这个目录塞进 sys.path，否则 Python 在 /app 下找不到同在网盘里的 gradio_sd3.py
if os.path.exists(model_dir) and model_dir not in sys.path:
    sys.path.insert(0, model_dir)

try:
    from gradio_sd3 import StableDiffusion3TryOnPipeline

    print("[FitDiT] 🎉 成功从网盘目录中本地导入 StableDiffusion3TryOnPipeline 类！")
except Exception as import_err:
    print(f"[FitDiT] ❌ 致命错误！引入源码脚本失败，详情: {str(import_err)}")
    StableDiffusion3TryOnPipeline = None

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[FitDiT] 正在初始化物理设备 {device}，开始读取网盘大权重...")

try:
    if StableDiffusion3TryOnPipeline is not None:
        # 传入 model_dir，让它在原地读取 model_index.json 和各组件权重
        pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        if device == "cuda":
            pipeline.to(device)
        print("[FitDiT] 🎉🎉 试衣大模型与自适应管线加载成功，Serverless 算力单元已全线就绪！")
    else:
        raise ValueError("核心依赖类 StableDiffusion3TryOnPipeline 缺失，容器终止。")
except Exception as init_err:
    print(f"[FitDiT] ❌ 致命错误！未能在初始化阶段成功加载管线: {str(init_err)}")
    pipeline = None


# ----------------------------------------------------
# 2. 图像编解码工具函数
# ----------------------------------------------------
def decode_base64_image(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[-1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")


def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


# ----------------------------------------------------
# 3. RunPod 主监听事件
# ----------------------------------------------------
def handler(job):
    try:
        if pipeline is None:
            return {"status": "failed", "error": "模型管线在容器初始化阶段加载失败，请核对云盘挂载结构。"}

        job_input = job.get('input', {})
        model_b64 = job_input.get('model_image')
        garment_b64 = job_input.get('garment_image')
        mask_b64 = job_input.get('mask_image')

        prompt = job_input.get('prompt', "a person wearing the garment, fashion, photorealistic")
        steps = int(job_input.get('steps', 30))
        guidance_scale = float(job_input.get('guidance_scale', 3.5))

        if not model_b64 or not garment_b64:
            return {"status": "failed", "error": "必须提供有效参数: model_image 且 garment_image"}

        model_image = decode_base64_image(model_b64)
        garment_image = decode_base64_image(garment_b64)
        original_size = model_image.size

        if mask_b64:
            mask_image = decode_base64_image(mask_b64)
        else:
            mask_image = Image.new("RGB", original_size, (255, 255, 255))

        target_size = (768, 1024)
        model_img_resized = model_image.resize(target_size)
        garment_img_resized = garment_image.resize(target_size)
        mask_img_resized = mask_image.resize(target_size)

        with torch.inference_mode():
            output = pipeline(
                prompt=prompt,
                image=model_img_resized,
                mask_image=mask_img_resized,
                garment_image=garment_img_resized,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            )

            if hasattr(output, "images") and len(output.images) > 0:
                generated_img = output.images
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                generated_img = output
            else:
                generated_img = output

            final_image = generated_img.resize(original_size)

        result_b64 = encode_image_to_base64(final_image)
        return {
            "status": "success",
            "result_image": result_b64
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": f"推理服务内部异常终止: {str(e)}"
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

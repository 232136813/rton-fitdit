import os
import sys
import torch
import runpod
import base64
from PIL import Image
from io import BytesIO

# ----------------------------------------------------
# 1. 强行注入云盘路径并手动引入自定义管线类 (最稳妥解法)
# ----------------------------------------------------
model_dir = "/runpod-volume/FitDiT"

if os.path.exists(model_dir):
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
else:
    print(f"[FitDiT] 警告：在 /runpod-volume/FitDiT 路径下未检测到云盘挂载！")

# 核心突破：绕过 diffusers 官方无法识别的 Bug，直接从云盘内的脚本文件里把类抢先 import 进来！
try:
    # 之前我们通过 ls -la 看到云盘里存在 gradio_sd3.py
    from gradio_sd3 import StableDiffusion3TryOnPipeline

    print("[FitDiT] 成功从本地脚本 gradio_sd3.py 中导入 StableDiffusion3TryOnPipeline 类。")
except Exception as import_err:
    print(f"[FitDiT] 引入自定义脚本失败，将尝试默认加载。详情: {str(import_err)}")
    StableDiffusion3TryOnPipeline = None

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[FitDiT] 正在初始化物理设备 {device}，开始加载试衣管线...")

try:
    if StableDiffusion3TryOnPipeline is not None:
        # 1. 如果手动引入类成功，直接显式调用它的 from_pretrained（最为精准安全）
        pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
    else:
        # 2. 备用兜底方案
        from diffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )

    if device == "cuda":
        pipeline.to(device)
    print("[FitDiT] 🎉 试衣模型与自适应管线加载成功，Serverless 算力单元已就绪！")
except Exception as init_err:
    print(f"[FitDiT] ❌ 致命错误！未能在初始化阶段成功加载管线: {str(init_err)}")
    pipeline = None


# ----------------------------------------------------
# 2. 图像编解码工具函数
# ----------------------------------------------------
def decode_base64_image(base64_str):
    """支持 Data URL 格式的多鲁棒性 Base64 解码"""
    if "," in base64_str:
        base64_str = base64_str.split(",")[-1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")


def encode_image_to_base64(image):
    """将 PIL 图像转换为高清晰度 JPEG Base64 字符串"""
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
            return {"status": "failed", "error": "模型管线在容器初始化阶段加载失败，请检查 /runpod-volume 挂载目录结构。"}

        # 解析请求包
        job_input = job.get('input', {})
        model_b64 = job_input.get('model_image')  # 模特原图 (Base64)
        garment_b64 = job_input.get('garment_image')  # 衣服平铺图 (Base64)
        mask_b64 = job_input.get('mask_image')  # 试衣选区遮罩图 (Base64)

        # 提取超参数 (设置默认值)
        prompt = job_input.get('prompt', "a person wearing the garment, fashion, photorealistic")
        steps = int(job_input.get('steps', 30))
        guidance_scale = float(job_input.get('guidance_scale', 3.5))

        if not model_b64 or not garment_b64:
            return {"status": "failed", "error": "必须提供有效参数: model_image 且 garment_image"}

        # 恢复图片对象并记录模特原始尺寸
        model_image = decode_base64_image(model_b64)
        garment_image = decode_base64_image(garment_b64)
        original_size = model_image.size

        # 处理遮罩 (Mask)
        if mask_b64:
            mask_image = decode_base64_image(mask_b64)
        else:
            mask_image = Image.new("RGB", original_size, (255, 255, 255))

        # 转换至 FitDiT 最佳性能分辨率 768x1024
        target_size = (768, 1024)
        model_img_resized = model_image.resize(target_size)
        garment_img_resized = garment_image.resize(target_size)
        mask_img_resized = mask_image.resize(target_size)

        # 启动算力推理
        with torch.inference_mode():
            output = pipeline(
                prompt=prompt,
                image=model_img_resized,
                mask_image=mask_img_resized,
                garment_image=garment_img_resized,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            )

            # 提取生成的 PIL 图片对象
            if hasattr(output, "images") and len(output.images) > 0:
                generated_img = output.images[0]
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                generated_img = output[0]
            else:
                generated_img = output

            # 将生成的图片无损重置回模特初始传入的比例与尺寸
            final_image = generated_img.resize(original_size)

        # 编码并组装返回结构
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

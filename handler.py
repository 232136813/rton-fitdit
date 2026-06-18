import os
import torch
import runpod
import base64
from PIL import Image
from io import BytesIO

# 从 diffusers 导入由 FitDiT 贡献的 StableDiffusion3 试衣管线
from diffusers import StableDiffusion3TryOnPipeline

# ----------------------------------------------------
# 1. 全局初始化（容器启动时执行，避免冷启动重复加载）
# ----------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "/models/FitDiT"

print(f"正在加载 FitDiT 试衣管线（设备: {device}）...")

# 加载官方权重，使用 bfloat16 优化显存并提升加载速度
pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16
)
pipeline.to(device)

print("FitDiT 试衣模型加载完毕，Serverless 准备就绪。")


# ----------------------------------------------------
# 2. 辅助工具函数
# ----------------------------------------------------
def decode_base64_image(base64_str):
    """将前端传入的 Base64 图片还原为 PIL 图像"""
    # 兼容带 Data URL 头和不带头的 base64 字符串
    if "," in base64_str:
        base64_str = base64_str.split(",")[-1]
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")


def encode_image_to_base64(image):
    """将生成的 PIL 图像转换成 Base64 字符串返回"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


# ----------------------------------------------------
# 3. RunPod 主处理程序
# ----------------------------------------------------
def handler(job):
    try:
        # 获取接口传入的 JSON 数据
        job_input = job.get('input', {})

        # 核心图片参数 (Base64)
        model_b64 = job_input.get('model_image')  # 模特图
        garment_b64 = job_input.get('garment_image')  # 衣服平铺图
        mask_b64 = job_input.get('mask_image')  # 试衣遮罩图（可选）

        # 超参数
        prompt = job_input.get('prompt', "a person wearing the garment")
        steps = int(job_input.get('steps', 30))
        guidance_scale = float(job_input.get('guidance_scale', 3.5))

        # 校验必备输入
        if not model_b64 or not garment_b64:
            return {"error": "缺少核心参数 'model_image' 或 'garment_image'"}

        # 转换图片格式
        model_image = decode_base64_image(model_b64)
        garment_image = decode_base64_image(garment_b64)

        # 处理试衣遮罩 (Mask)
        # 如果调用方未传入预生成的 Mask，FitDiT 提供了依据图片全刷或默认蒙版策略
        # 强烈建议在前端或借助额外的插件完成精准 Mask 提取以达到最佳效果
        if mask_b64:
            mask_image = decode_base64_image(mask_b64)
        else:
            # 备用方案：生成一张与模特图同等大小的全白蒙版，交由模型自行解构
            mask_image = Image.new("RGB", model_image.size, (255, 255, 255))

        # 执行深度学习推理
        with torch.inference_mode():
            # 统一调整分辨率至模型推荐尺寸 (1024x768 或等比例)
            model_image_resized = model_image.resize((768, 1024))
            garment_image_resized = garment_image.resize((768, 1024))
            mask_image_resized = mask_image.resize((768, 1024))

            output_images = pipeline(
                prompt=prompt,
                image=model_image_resized,
                mask_image=mask_image_resized,
                garment_image=garment_image_resized,
                num_inference_steps=steps,
                guidance_scale=guidance_scale
            ).images

            # 还原输出为模特原图分辨率大小
            result_image = output_images[0].resize(model_image.size)

        # 编码返回
        result_b64 = encode_image_to_base64(result_image)
        return {
            "status": "success",
            "result_image": result_b64
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


# 启动 RunPod Worker 监听
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

import os
import torch
import runpod
import base64
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusion3TryOnPipeline

# ----------------------------------------------------
# 1. 容器全局初始化 (单例模式防冷启动耗时)
# ----------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_dir = "/models/FitDiT"

print(f"[FitDiT] 开始在设备 {device} 上加载试衣管线...")

# 加载官方权重，使用 float16 保证在 24G 显存（如 RTX 4090）上高效运行
pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.float16
)
pipeline.to(device)

print("[FitDiT] 试衣模型加载完成，Serverless 接收端已就绪。")


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
        # 解析请求包
        job_input = job.get('input', {})
        model_b64 = job_input.get('model_image')  # 模特原图 (Base64)
        garment_b64 = job_input.get('garment_image')  # 衣服平铺图 (Base64)
        mask_b64 = job_input.get('mask_image')  # 试衣选区遮罩图 (Base64)

        # 提取超参数 (设置默认值)
        prompt = job_input.get('prompt', "a person wearing the garment, fashion, photorealistic")
        steps = int(job_input.get('steps', 30))
        guidance_scale = float(job_input.get('guidance_scale', 3.5))

        # 拦截无效输入
        if not model_b64 or not garment_b64:
            return {"status": "failed", "error": "必须提供参数: model_image 且 garment_image"}

        # 恢复图片对象
        model_image = decode_base64_image(model_b64)
        garment_image = decode_base64_image(garment_b64)

        # 记录模特原始尺寸，方便最终将结果图还原输出
        original_size = model_image.size

        # 处理遮罩 (Mask)
        if mask_b64:
            mask_image = decode_base64_image(mask_b64)
        else:
            # 如果未传入 Mask，默认生成全白遮罩让模型自行全画幅泛化
            mask_image = Image.new("RGB", original_size, (255, 255, 255))

        # 核心：FitDiT 最佳性能分辨率是 768x1024
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
            # 获取模型输出图
            generated_img = output.images[0]

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
            "error": f"推理服务异常终止: {str(e)}"
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

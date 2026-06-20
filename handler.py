import os
import sys
import torch
import runpod
import base64
from PIL import Image
from io import BytesIO

# ----------------------------------------------------
# 1. 动态注入模型源码路径与全局初始化
# ----------------------------------------------------
model_dir = "/runpod-volume/FitDiT"

# 确保 diffusers 在加载 trust_remote_code 时能顺利在模型根目录下找到与其配套的本地 python 模块
if os.path.exists(model_dir) and model_dir not in sys.path:
    sys.path.insert(0, model_dir)

from diffusers import DiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[FitDiT] 正在初始化物理设备 {device}，尝试从目标路径 [{model_dir}] 加载试衣管线...")

try:
    # 完美适配固定版本生态，开启 trust_remote_code 自动调用本地权重目录下的自定义类
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
    print(f"[FitDiT] 提示：请核对 RunPod 云网盘是否挂载正确，且 /models/FitDiT 下是否存在 model_index.json 及其关联脚本。")
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
        # 拦截模型未初始化成功的边界情况
        if pipeline is None:
            return {"status": "failed", "error": "模型管线在容器初始化阶段加载失败，请检查挂载路径与权重文件结构。"}

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
            return {"status": "failed", "error": "必须提供有效参数: model_image 且 garment_image"}

        # 恢复图片对象并记录模特原始尺寸，方便最终将结果图还原输出
        model_image = decode_base64_image(model_b64)
        garment_image = decode_base64_image(garment_b64)
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

            # 鲁棒性提取：完美兼容对象返回或列表返回，防止在不同 diffusers 版本间切换时崩溃
            if hasattr(output, "images"):
                generated_img = output.images[0]
            elif isinstance(output, (list, tuple)):
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

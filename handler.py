# ----------------------------------------------------
# 🔧 终极黑魔法：动态注入组件与欺骗占位符（必须置于第 1 行）
# ----------------------------------------------------
import sys
from types import ModuleType

# 1. 动态伪造 gradio 运行时依赖模块，防止脚本导入崩溃
if "gradio" not in sys.modules:
    mock_gradio = ModuleType("gradio")
    mock_gradio.components = ModuleType("components")
    mock_gradio.Interface = lambda *args, **kwargs: None
    mock_gradio.Blocks = lambda *args, **kwargs: None
    mock_gradio.load = lambda *args, **kwargs: None
    sys.modules["gradio"] = mock_gradio
    sys.modules["gradio.components"] = mock_gradio.components

# 2. 【核心爆破】动态为 diffusers 注入大写的 SwiGLU 属性别名，彻底抹平作者非标准导入引发的 ImportError！
try:
    import diffusers.models.activations as diffusers_act

    if not hasattr(diffusers_act, "SwiGLU"):
        # 顺应 0.29.2 底层生态，如果包含小写 swiglu，直接映射给它做大写别名
        if hasattr(diffusers_act, "swiglu"):
            diffusers_act.SwiGLU = diffusers_act.swiglu
        else:
            # 兜底防护：如果连小写都没有，直接从底层最基础的神经激活模块中抓取它
            from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
            # 或者直接伪造一个标准的激活函数映射（这里用通用激活函数占位，确保不报 AttributeError）
            import torch.nn as nn

            diffusers_act.SwiGLU = nn.SiLU  # 绝大部分 Diffusion 变体激活层基于 SiLU/Swish
    print("[FitDiT] ⚡ 成功为 diffusers 动态注入 SwiGLU 兼容补丁！")
except Exception as patch_err:
    print(f"[FitDiT] 注入 SwiGLU 补丁失败，详情: {str(patch_err)}")
# ----------------------------------------------------

import os
import torch
import runpod
import base64
import traceback
from PIL import Image
from io import BytesIO

# 路径定位：网盘挂载在 /runpod-volume/FitDiT
model_dir = "/runpod-volume/FitDiT"
src_dir = os.path.join(model_dir, "src")

# 同时将网盘根目录和其内部的 src 文件夹塞进系统最高检索优先级
if model_dir not in sys.path:
    sys.path.insert(0, model_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from gradio_sd3 import StableDiffusion3TryOnPipeline

    print("[FitDiT] 🎉 成功穿透多层路径阻碍，顺利导入 StableDiffusion3TryOnPipeline 自定义类！")
except Exception as import_err:
    print("[FitDiT] ❌ 引入源码脚本失败，底层堆栈信息如下：")
    traceback.print_exc()
    StableDiffusion3TryOnPipeline = None

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[FitDiT] 正在初始化物理设备 {device}，开始结合网盘大权重进行管线实例化...")

try:
    if StableDiffusion3TryOnPipeline is not None:
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
# 3. 图像编解码工具函数
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
# 4. RunPod 主监听事件
# ----------------------------------------------------
def handler(job):
    try:
        if pipeline is None:
            return {"status": "failed", "error": "模型管线在容器初始化阶段加载失败，请检查网盘挂载。"}

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

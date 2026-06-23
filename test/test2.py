wm_bytes = b'!\x88\x06 '

# 1. 打印出精确的 32 位二进制 01 串
binary_str = ''.join(f'{b:08b}' for b in wm_bytes)
print(f"【核心验证】这幅图的 32 位二进制数字指纹为: \n{binary_str}\n")

# 2. 尝试用市面上各大厂商通用的编码格式（如 Latin-1）去硬解它
try:
    decoded_text = wm_bytes.decode('latin-1')
    print(f"尝试 Latin-1 解码文本: {decoded_text}")
except Exception:
    pass

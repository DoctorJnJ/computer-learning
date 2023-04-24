import math
"""
输入数据大小为 (batch_size, channels, height, width)
卷积核大小为 (kernel_size, kernel_size)
填充大小为 (padding, padding)
步长为 (stride, stride)：
"""
batch_size = eval(input())
channels = eval(input())
height = eval(input())
width = eval(input())
kernel_size = eval(input())
padding = eval(input())
stride = eval(input())

def conv_output_size(input_size, kernel_size, padding, stride):
    """
    计算卷积后的输出大小
    """
    output_size = math.floor((input_size + 2 * padding - kernel_size) / stride) + 1
    return output_size

# 计算卷积后的输出大小
out_height = conv_output_size(height, kernel_size, padding, stride)
out_width = conv_output_size(width, kernel_size, padding, stride)

# 输出结果
print("输入大小：", (batch_size, channels, height, width))
print("卷积核大小：", (kernel_size, kernel_size))
print("填充大小：", (padding, padding))
print("步长大小：", (stride, stride))
print("输出大小：", (batch_size, channels, out_height, out_width))
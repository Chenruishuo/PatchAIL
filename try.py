import torch
from dm_control.suite.wrappers import action_scale, pixels

# 假设你有一个形状为 (batch_size, num_classes) 的tensor
# 这里为了演示目的，假设 batch_size=1，num_classes=5
input_tensor = torch.randn(2, 5)
print("原始输入:", input_tensor)
# 对最后一维进行softmax操作
softmax_output = torch.nn.functional.softmax(input_tensor, dim=-1)
print("Softmax输出:", softmax_output)
# 从softmax输出中采样一个值
sampled_index = torch.multinomial(softmax_output, 1)
print("采样处索引:", sampled_index)
# 输出采样处的softmax值
sampled_probability = torch.gather(softmax_output, -1, sampled_index)




print("采样处的Softmax值:", sampled_probability)
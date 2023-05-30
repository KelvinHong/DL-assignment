from utils import *
m = SingleLayerCAM()
m.eval()
inp = torch.randn((10,3,256,256), requires_grad=True)
cams = m.get_cam(inp)
print(cams.min(), cams.max())
# x = torch.tensor(2., requires_grad=True)
# y = x**2+1
# y.backward()
# print(x.grad)
# x.grad.zero_()
# y.backward()
# print(x.grad)
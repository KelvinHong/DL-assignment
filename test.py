from utils import *
x = torch.randn((2,3,256,256))
m = baseCAM()
m.get_detail_cam(x)
# x = torch.tensor(2., requires_grad=True)
# y = x**2+1
# y.retain_grad()
# z = torch.exp(y)-10
# z.backward(retain_graph=True)
# print(x.grad)
# print(y.grad)
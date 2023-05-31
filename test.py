from utils import *
x = torch.randn((2,3,256,256))
m = baseCAM()
m.get_detail_cam(x)
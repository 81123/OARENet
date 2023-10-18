import torch.nn as nn
import sys
sys.path.append(r'C:\Users\Rain\Desktop\four_one\code\ROADecoder\GAMSNet_OA')
from networks.backbone import build_ResNet
from networks.decoder import build_decoder,build_decoder2
from networks.intersection import build_erase
import torch
from networks.dinknet import BAM_LinkNet50,BAM_LinkNet50_T,LinkNet50_T
# from networks.swin_transformer import SwinTransformer

class GAMSNet_OAM(nn.Module):
    def __init__(self):
        super(GAMSNet_OAM, self).__init__()
        self.erase_channel=2
        self.resnet = BAM_LinkNet50_T()
        self.decoder = build_decoder()
        self.erase = build_erase(erase_channel=self.erase_channel)
        self.deconv = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )
        self.finalconv = nn.Conv2d(64+self.erase_channel,1,1)
    def forward(self,x):
        e1, e2, e3, e4=self.resnet(x)
        x = self.decoder(e1, e2, e3, e4)
        x1 = self.erase(x)
        x2 = self.deconv(x)
        x2 = torch.cat((x1, x2),dim=1)
        x = self.finalconv(x2)
        return torch.sigmoid(x)
if __name__ == "__main__":
    import torch
    from tqdm import tqdm
    from thop import profile
    model = GAMSNet_OAM().cuda()
    input = torch.rand(2, 3, 256, 256).cuda()

    iterations = 50
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # GPU预热
    for _ in tqdm(range(20)):
        _ = model(input)

    # 测速
    times = torch.zeros(iterations)  # 存储每轮iteration的时间
    with torch.no_grad():
        for iter in tqdm(range(iterations)):
            starter.record()
            _ = model(input)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 计算时间
            times[iter] = curr_time
            # print(curr_time)

    mean_time = times.mean().item()
    print("Inference time: {:.6f}s, FPS: {} ".format(mean_time/1000, 1000 / mean_time))
    flops,params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
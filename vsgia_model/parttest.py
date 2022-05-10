import torch

if __name__ == '__main__':

    model_gf=torch.load("/home/nku120/Desktop/VSG-IApublish/modelparas/vsgia_0.923-0.128-0.0691.pth.tar")

    torch.save(model_gf["state_dict"],"/home/nku120/Desktop/VSG-IApublish/modelparas/model_gazefollow.pt")
    #
    model_vat=torch.load("/home/nku120/Desktop/VSG-IApublish/modelparas/vsgia_0.880-0.118-0.881.pth.tar")

    torch.save(model_vat['state_dict'],"/home/nku120/Desktop/VSG-IApublish/modelparas/model_videotargetattention.pt")
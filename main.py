import torch
from torchvision import transforms
from dataloader import get_loader
from train import Trainer
#from eval import Eval



def main():
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    trainloader = get_loader('train', train_transform)
    testloader = get_loader('test', test_transform)
    trainer = Trainer(trainloader, testloader)

    #trainer.train_classifier()
    #trainer.qualitative_evaluation()
    # if 1= enhance, 2 = adversarial.
    # second term denotes the target classifiers.
    #for testing black box setting, set ImgNetPretrained=True in Discriminator_cls
    trainer.train_adversarial(1, 5)



if __name__ == "__main__":
    main()

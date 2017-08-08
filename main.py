import torch
from torchvision import transforms
from dataloader import get_loader
from train import Trainer
from eval import Eval

def main():
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    trainloader = get_loader('train', train_transform)
    testloader = get_loader('test', test_transform)
    trainer = Trainer(trainloader, testloader)
    trainer.train_classifier()
    trainer.train_adversarial()
    # trainer.train_conv_mask()
    # Eval.eval()

if __name__ == "__main__":
    main()

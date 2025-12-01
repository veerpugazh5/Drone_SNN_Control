"""
Binary SNN: Turning vs Straight
Simpler 2-class output
"""
import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x > 0).float()
    
    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        grad = alpha / (2 * (1 + (alpha * x).pow(2)))
        return grad_out * grad, None


def spike(x):
    return SurrogateSpike.apply(x)


class LIF(nn.Module):
    def __init__(self, beta=0.5, threshold=1.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
    
    def forward(self, x, mem):
        mem = self.beta * mem + x
        spk = spike(mem - self.threshold)
        mem = mem - spk * self.threshold
        return spk, mem


class DroneSNNBinary(nn.Module):
    """
    Binary SNN for Turning vs Straight
    Input: (B, 2, 64, 64) -> Output: (B, 2)
    """
    def __init__(self, num_steps=10, beta=0.5, dropout=0.2):
        super().__init__()
        self.num_steps = num_steps
        
        # Conv block 1: 2 -> 32, 64 -> 32
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.lif1 = LIF(beta=beta)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(dropout)
        
        # Conv block 2: 32 -> 64, 32 -> 16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = LIF(beta=beta)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(dropout)
        
        # Conv block 3: 64 -> 128, 16 -> 8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lif3 = LIF(beta=beta)
        self.pool3 = nn.MaxPool2d(2)
        
        # FC: 128*8*8 -> 2
        self.fc = nn.Linear(128 * 8 * 8, 2)
        self.lif_out = LIF(beta=beta)
    
    def forward(self, x, return_hidden=False):
        B = x.size(0)
        device = x.device
        
        mem1 = torch.zeros(B, 32, 64, 64, device=device)
        mem2 = torch.zeros(B, 64, 32, 32, device=device)
        mem3 = torch.zeros(B, 128, 16, 16, device=device)
        mem_out = torch.zeros(B, 2, device=device)
        
        out_spikes = []
        hidden_spikes = [] if return_hidden else None
        
        for _ in range(self.num_steps):
            cur1 = self.bn1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.drop1(self.pool1(spk1))
            
            cur2 = self.bn2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.drop2(self.pool2(spk2))
            
            cur3 = self.bn3(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.pool3(spk3)
            
            if return_hidden:
                hidden_spikes.append(spk3)
            
            flat = spk3.view(B, -1)
            cur_out = self.fc(flat)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            out_spikes.append(spk_out)
        
        out = torch.stack(out_spikes, dim=1).mean(dim=1)
        
        if return_hidden:
            return out, torch.stack(hidden_spikes, dim=1)
        return out


if __name__ == "__main__":
    model = DroneSNNBinary()
    x = torch.randn(4, 2, 64, 64)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")




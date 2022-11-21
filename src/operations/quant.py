from numpy import uint32
import torch


class FPQuant(torch.autograd.Function):
    @staticmethod
    def __to_exponent_mantissa_width__(
        inp: torch.Tensor,
        max_log: torch.Tensor,
        mantissa_bitwidth: int,
        min_mantissa: int,
        max_mantissa: int,
    ):
        shp = inp.shape
        # NOTE THAT THIS IS -1 BECAUSE OF THE LEADING 1 IN 1.b0b1b2b3...*2^E
        exponent_needed = (mantissa_bitwidth - max_log - 1) * torch.ones(
            shp, device=inp.device
        )
        first_mant_w = torch.pow(2, exponent_needed)
        inp = inp * first_mant_w
        # Half LSB Rounding:
        inp = torch.round(inp)
        inp = torch.clamp(inp, min=min_mantissa, max=max_mantissa)
        inp = inp / first_mant_w
        return inp

    @staticmethod
    def __find_exponent__(inp: torch.Tensor, min_exponent: int, max_exponent: int):
        absolute = torch.abs(inp)
        value_log = torch.log2(absolute)
        value_log = torch.clamp(value_log, min_exponent, max_exponent)
        int_log = torch.floor(value_log)
        return int_log

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        min_e: int,
        max_e: int,
        mantissa_bit: int,
        min_m: int,
        max_m: int,
    ):
        max_exponent = FPQuant.__find_exponent__(inp, min_e, max_e)
        quantized_act = FPQuant.__to_exponent_mantissa_width__(
            inp, max_exponent, mantissa_bit, min_m, max_m
        )
        return quantized_act

    @staticmethod
    def backward(ctx, grad):
        # STE Gradient
        return grad, None, None, None, None, None


fpquant = FPQuant.apply


class FPQuantize(torch.nn.Module):
    def __init__(self, mantissa=4) -> None:
        super().__init__()
        self.e = 7
        self.m = mantissa

    def extra_repr(self):
        return f"e={self.e},m={self.m}"

    def forward(self, tens):
        max = 2 ** (self.e - 1) - 1
        min = -(2 ** (self.e - 1))
        min_m = -(2 ** self.m) + 1
        max_m = (2 ** self.m) - 1
        return fpquant(tens, min, max, self.m, min_m, max_m)


if __name__ == "__main__":
    from numpy import int32

    mantissa = 4
    a = torch.rand([64, 64, 3, 3]) - 0.5
    b = FPQuantize(mantissa)(a)
    print(a[0, 0])
    print(b[0, 0])
    print(a[:, 0, 0, 0])
    print(b[:, 0, 0, 0])
    print(a[:, 0, 0, 0] - b[:, 0, 0, 0])
    print(f"{a[0,0,0,0].numpy().view(uint32):032b}")
    print("s" + "e" * 8 + "m" * mantissa)
    print(f"{b[0,0,0,0].numpy().view(uint32):032b}")

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Parameter
import torch.nn.functional as F
from .dropout import DropMask, createMask

class CustomLSTM(nn.Module):
    def __init__(self, model_params):
        
        super(CustomLSTM, self).__init__()

        input_size = model_params["input_size"]
        hidden_size = model_params["hidden_size"]
        output_size = model_params["output_size"]
        number_static_predictors = model_params["number_static_predictors"]

        self.fc0 = nn.Linear(input_size + number_static_predictors, hidden_size)

        self.lstm = nn.LSTM(hidden_size , hidden_size, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 64)

        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x, static_params):

        x_ds = torch.cat(
             (x,
              static_params.unsqueeze(1).repeat(1, x.size(1), 1)),
              dim=-1,
         )

        l1 = torch.relu(self.fc0(x_ds))   

        lstm_output, _ = self.lstm(l1)

        out =  self.fc2(torch.relu(self.fc1(lstm_output)))


        return out




class CudnnLSTMlyr(nn.Module):
    def __init__(self, *, inputSize, hiddenSize, dr=0.5, drMethod="drW", gpu=0):
        super(CudnnLSTM, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]
        self.cuda()
        self.name = "CudnnLstm"
        self.is_legacy = True

        self.reset_mask()
        self.reset_parameters()

    def _apply(self, fn):
        ret = super(CudnnLSTM, self)._apply(fn)
        return ret

    def __setstate__(self, d):
        super(CudnnLSTM, self).__setstate__(d)
        self.__dict__.setdefault("_data_ptrs", [])
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        if isinstance(self._all_weights[0][0], str):
            return
        self._all_weights = [["w_ih", "w_hh", "b_ih", "b_hh"]]

    def reset_mask(self):
        self.maskW_ih = createMask(self.w_ih, self.dr)
        self.maskW_hh = createMask(self.w_hh, self.dr)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None, cx=None, doDropMC=False, dropoutFalse=False):
        # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
        if dropoutFalse and (not doDropMC):
            doDrop = False
        elif self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = input.size(1)

        if hx is None:
            hx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)
        if cx is None:
            cx = input.new_zeros(1, batchSize, self.hiddenSize, requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight = [
                DropMask.apply(self.w_ih, self.maskW_ih, True),
                DropMask.apply(self.w_hh, self.maskW_hh, True),
                self.b_ih,
                self.b_hh,
            ]
        else:
            weight = [self.w_ih, self.w_hh, self.b_ih, self.b_hh]

        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
        # input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
        # self.hiddenSize, 1, False, 0, self.training, False, (), None)
        if torch.__version__ < "1.8":
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hiddenSize,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        else:
            output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
                input,
                weight,
                4,
                None,
                hx,
                cx,
                2,  # 2 means LSTM
                self.hiddenSize,
                0,
                1,
                False,
                0,
                self.training,
                False,
                (),
                None,
            )
        return output, (hy, cy)

    @property
    def all_weights(self):
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]




class CudnnLSTM(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, warmUpDay=None):
        super(CudnnLSTM, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        
        self.linearIn = torch.nn.Linear(nx, hiddenSize)

        self.lstm = CudnnLSTMlyr(inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        # self.gpu = 1
        self.name = "CudnnLstmModel"
        self.is_legacy = True
        # self.drtest = torch.nn.Dropout(p=0.4)
        self.warmUpDay = warmUpDay

    def forward(self, x1, x2, doDropMC=False, dropoutFalse=False):
        """

        :param inputs: a dictionary of input data (x and potentially z data)
        :param doDropMC:
        :param dropoutFalse:
        :return:
        """
        x2 = x2.unsqueeze(1).expand(-1, x1.shape[1], -1)
        x = torch.cat((x1, x2), -1)

        if not self.warmUpDay is None:
            x, warmUpDay = self.extend_day(x, warmUpDay=self.warmUpDay)

        x0 = F.relu(self.linearIn(x))

        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse)
        # outLSTMdr = self.drtest(outLSTM)
        out = self.linearOut(outLSTM)

        if not self.warmUpDay is None:
            out = self.reduce_day(out, warmUpDay=warmUpDay)

        return out

    def extend_day(self, x, warm_up_day):
        x_num_day = x.shape[0]
        warm_up_day = min(x_num_day, warm_up_day)
        x_select = x[:warm_up_day, :, :]
        x = torch.cat([x_select, x], dim=0)
        return x, warm_up_day

    def reduce_day(self, x, warm_up_day):
        x = x[warm_up_day:, :, :]
        return x

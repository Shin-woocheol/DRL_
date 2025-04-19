import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscretePolicy(nn.Module): # hidden_dim을 tuple로 설정해서 hidden layer 각각 dim달라지게도 설정 가능.
    def __init__(self, state_dim, action_dim, hidden_dims = (128, ), activation_func = F.relu): # discrete에서의 action dim은 정말 possible action ex) up, down, left, right
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0]) # (batch x state_dim) -> (batch x hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) -1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1]) #tuple 쓰인대로 설정. # (batch x hidden_dims[i]) -> (batch x hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim) #possible action에 대한 확률값들 # (batch x hidden_dims[-1]) -> (batch x action_dim)
        self.activation_func = activation_func #다른 함수에서 사용하기 위해.

    def forward(self, state): #model operation
        x = self.activation_func(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation_func(hidden_layer(x))
        prob = F.softmax(self.output_layer(x), dim = -1) #output layer지나면 (batch_size x action_dim)고로 -1로 softmax해줌.
        # discrete의 경우에는 할 수 있는 action 전부를 probability로 나타냄. action 이 finite
        return prob

class GaussianPolicy(nn.Module): #* Discrete policy와 다른 부분은, continous action이기에, action distribution을 gaussian으로 가정하고, nn으로부터는 mu와 sigma를 output으로 받기.
    def __init__(self, state_dim, action_dim, hidden_dims = (128, ), activation_func = F.relu): # continous의 action dim은 선택해줘야 하는 action의 갯수 ex) 가속도, 각도 면 2
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) -1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1]) #tuple 쓰인대로 설정.
            self.hidden_layers.append(hidden_layer)
        # self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.mu_layer = nn.Linear(hidden_dims[-1], action_dim) #action의 갯수만큼 mu 필요.
        self.log_sigma_layer = nn.Linear(hidden_dims[-1], action_dim) #* log sigma로 진행하는 이유는 sigma가 양수임을 보장하기 위함. x가 어떤 값이든 exp(x)는 양수임.
        self.activation_func = activation_func #다른 함수에서 사용하기 위해.

    def forward(self, state): #model operation
        x = self.activation_func(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation_func(hidden_layer(x))
        mu = self.mu_layer(x)
        log_sigma = self.log_sigma_layer(x) 
        return mu, log_sigma.exp()

class StateValue(nn.Module): #* baseline으로 사용하기 위한 state value function의 nn을 이용한 approximation
    def __init__(self, state_dim, hidden_dims = (128, ), activation_func = F.relu):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) -1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.activation_func = activation_func

    def forward(self, state): #model operation
        x = self.activation_func(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = self.activation_func(hidden_layer(x))
        state_value = self.output_layer(x)
        return state_value # (batch_size, 1)
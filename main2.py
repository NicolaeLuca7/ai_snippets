#%%
from time import sleep
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numbers as mp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


class plot_error_surfaces(object):
    
    # Constructor
    def __init__(self, w_range, b_range, X, Y, n_samples = 30, go = True):
        W = np.linspace(-w_range, w_range, n_samples)
        B = np.linspace(-b_range, b_range, n_samples)
        w, b = np.meshgrid(W, B)    
        Z = np.zeros((30,30))
        count1 = 0
        self.y = Y.numpy()
        self.x = X.numpy()
        for w1, b1 in zip(w, b):
            count2 = 0
            for w2, b2 in zip(w1, b1):
                Z[count1, count2] = np.mean((self.y - w2 * self.x + b2) ** 2)
                count2 += 1
            count1 += 1
        self.Z = Z
        self.w = w
        self.b = b
        self.W = []
        self.B = []
        self.LOSS = []
        self.n = 0
        if go == True:
            plt.figure()
            plt.figure(figsize = (7.5, 5))
            plt.axes(projection='3d').plot_surface(self.w, self.b, self.Z, rstride = 1, cstride = 1,cmap = 'viridis', edgecolor = 'none')
            plt.title('Cost/Total Loss Surface')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.show()
            plt.figure()
            plt.title('Cost/Total Loss Surface Contour')
            plt.xlabel('w')
            plt.ylabel('b')
            plt.contour(self.w, self.b, self.Z)
            plt.show()
    
    # Setter
    def set_para_loss(self, W, B, loss):
        self.n = self.n + 1
        self.W.append(W)
        self.B.append(B)
        self.LOSS.append(loss)
    
    # Plot diagram
    def final_plot(self): 
        ax = plt.axes(projection = '3d')
        ax.plot_wireframe(self.w, self.b, self.Z)
        ax.scatter(self.W,self.B, self.LOSS, c = 'r', marker = 'x', s = 200, alpha = 1)
        plt.figure()
        plt.contour(self.w,self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()
    
    # Plot diagram
    def plot_ps(self):
        plt.subplot(121)
        plt.ylim
        plt.plot(self.x, self.y, 'ro', label="training points")
        plt.plot(self.x, self.W[-1] * self.x + self.B[-1], label = "estimated line")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim((-10, 15))
        plt.title('Data Space Iteration: ' + str(self.n))

        plt.subplot(122)
        plt.contour(self.w, self.b, self.Z)
        plt.scatter(self.W, self.B, c = 'r', marker = 'x')
        plt.title('Total Loss Surface Contour Iteration' + str(self.n))
        plt.xlabel('w')
        plt.ylabel('b')
        plt.show()



class Dataset(Dataset):
    def __init__(self,x_data,y_data):
        self.x_data=x_data
        self.y_data=y_data
        self.len=x_data.shape[0]

    def __getitem__(self,id):
        return x_data[id],y_data[id]
    
    def __len__(self):
        return self.len


x_data=torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])#torch.arange(-3,3,1).view(-1,1)
x_data=x_data.to(torch.float32)

f = 1 * x_data - 3
y_data=f+0.2*torch.randn(x_data.size())

dataset=Dataset(x_data=x_data,y_data=y_data)
dataloader=DataLoader(dataset=dataset,batch_size=2)



class LR(nn.Module):
    def __init__(self,input_features,output_features):
        super(LR,self).__init__()
        rd=torch.tensor([[50.0,50.0]]).numpy()
        self.w=torch.tensor(rd,requires_grad=True)
        rd=torch.tensor([50.0]).view(-1,1).numpy()
        self.b=torch.tensor(rd,requires_grad=True)
        
        self.layer=nn.Linear(input_features,output_features)
        self.layer.state_dict()['weight'].data[0]=self.w
        self.layer.state_dict()['bias'].data[0]=self.b

        self.criterion=nn.MSELoss()
        
    def forward(self,x_data):
        return self.layer(x_data)

    #def getloss(self,py_data, y_data):
     #   return torch.mean((py_data-y_data)**2)

    def setweight(self,w):
        self.layer.state_dict()['weight'].data[0]=w

    def setbias(self,b):
        self.layer.state_dict()['bias'].data[0]=b

model=LR(2,1)
optmizer=optim.SGD(model.parameters(),0.1)

print(model(x_data))




LOSS=[]
for epoch in range(30):
    sum=0
    for x,y in dataloader:

        py=model(x)

        loss=model.criterion(py, y)

        optmizer.zero_grad()

        loss.backward()
        optmizer.step()

        sum+=loss.item()

    LOSS.append(sum)

plt.plot(LOSS,label = "Stochastic Gradient Descent")
plt.xlabel('iteration')
plt.ylabel('Cost/ total loss')

print("Final weight: ",model.layer.weight.item())
print("Final bias: ",model.layer.bias.item())












# %%

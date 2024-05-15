import pdb
import numpy as np
import torch
import os
import torch.nn.functional as F
from torch import nn
import yaml
#p prompt_memory.memory.grad
class Memory(nn.Module):
    def __init__(self,top_k=16,memory_size=256,feature_shape=(1,512),prompt_shape=(64,2048),lr=1e-2,gamma=1e-2,memory_dir="memory",device="cuda",default_ctx=None,Load_dir=None):
        super().__init__()
        if(Load_dir!=None):
            self.Load(Load_dir,device=device)
            return
        self.top_k=top_k
        self.memory_size=memory_size
        self.feature_shape=feature_shape
        self.lr=lr
        self.gamma=gamma
        self.memory_dir=memory_dir.strip('\\').strip('/')
        self.prompt_shape=prompt_shape
        self.device=device
        try:
            os.mkdir(memory_dir)
        except:
            pass
        self.keys=nn.Parameter(torch.zeros((self.memory_size,self.feature_shape[0],self.feature_shape[1]),dtype=torch.float32,requires_grad=True,device=device))
        # torch.zeros(self.memory_size,self.feature_shape*2,dtype=torch.float32)
        if(default_ctx==None):
            self.memory=nn.Parameter(torch.zeros(self.memory_size,prompt_shape[0],prompt_shape[1],dtype=torch.float32,requires_grad=True,device=device))
        else:
            self.memory = nn.Parameter(default_ctx.unsqueeze(0).expand(self.memory_size, -1, -1).to(device=device,dtype=torch.float32),requires_grad=True)
        self.loss_record=torch.zeros(self.memory.shape,device=self.device)
        self.tot=0
        # pdb.set_trace()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
    
    def Insert(self,feature):
        fea_=feature.reshape(self.feature_shape,-1).detach()
        mean=torch.mean(fea_,axis=1)
        std=torch.std(fea_,axis=1)
        key=torch.cat((mean,std),dim=0)
        # pth='{:04d}'.format(tot)+".pt"
        # torch.save(prompt,self.memory_dir+"/"+pth)
        # prompt=torch.zeros(self.prompt_shape,dtype=torch.float32)
        self.keys[self.tot].copy_(key)
        # self.memory[self.tot].copy_(prompt.clone())
        self.tot+=1
    
    def Augmentation(self,features_):
        features=features_.detach().clone()
        shape=features.shape
        pshape=self.prompt_shape
        
        if(len(shape)==1):
            features=features.reshape(1,shape[0]).detach()
            shape=features.shape
            
        ctx= torch.zeros((shape[0],pshape[0],pshape[1]),device=self.device)
        # pdb.set_trace()
        for i,fea in enumerate(features):
            if(self.tot<self.memory_size):
                self.Insert(fea)
            
            key=fea.reshape(self.feature_shape,-1).detach()
            
            # print(key.device,self.keys[:self.tot].device)
            key_sim=F.cosine_similarity(key,self.keys[:self.tot])
            top_sim, top_ids = torch.topk(key_sim, k=min(self.top_k,self.tot))
            top_sim[top_sim<0]=0.
            top_sim=top_sim/top_sim.sum()
            for j,sim in enumerate(top_sim):
                id=top_ids[j]
                # pdb.set_trace()
                ctx[i]+=sim*self.memory[id]
                # self.keys[id]+=sim*self.gamma*(key-self.keys[id])
                
        return ctx
    
    def forward(self,features_):
        return self.Augmentation(features_)
    
    def Step(self):
        tmp=-self.memory.detach().cpu().clone()
        self.optimizer.step()
        tmp+=self.memory.detach().cpu().clone()
        self.loss_record+=tmp.to(device=self.device)
        # print(self.memory.grad)
        self.optimizer.zero_grad()

    def Save(self,dir=None):
        if(dir==None):
            dir=self.memory_dir
        data={
            "top_k":self.top_k,
            "memory_size":self.memory_size,
            "feature_shape":self.feature_shape,
            "prompt_shape":self.prompt_shape,
            "lr":self.lr,
            "gamma":self.gamma,
            "memory_dir":self.memory_dir,
            "tot":self.tot,
        }
        with open(self.memory_dir+'/cfg.yml', 'w') as file:
            yaml.dump(data, file)
        torch.save([self.keys.detach().cpu(),self.memory.detach().cpu()],self.memory_dir+"/memory.pt")
        torch.save(self.loss_record.cpu(),self.memory_dir+"/loss_record.pt")
        
    def Load(self,dir="memory",device="cuda"):
        file_path = dir+'/cfg.yml'
        with open(file_path, 'r') as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            
        self.top_k=data['top_k']
        self.memory_size=data['memory_size']
        self.feature_shape=data['feature_shape']
        self.lr=data['lr']
        self.gamma=data['gamma']
        self.memory_dir=data['memory_dir'].strip('\\').strip('/')
        self.prompt_shape=data['prompt_shape']
        self.device=device
        self.tot=data['tot']
        if os.path.exists(self.memory_dir+"/memory.pt"):
            self.keys,self.memory=torch.load(self.memory_dir+"/memory.pt")
            # pdb.set_trace()
            self.keys=self.keys.detach().cpu()
            self.keys=self.keys.to(device)
            self.keys.requires_grad_(True)
            self.memory=self.memory.to(device)
            self.memory.requires_grad_(True)
        else:
            self.memory=nn.Parameter(torch.zeros(self.memory_size,self.prompt_shape[0],self.prompt_shape[1],dtype=torch.float32,requires_grad=True,device=device))
            self.keys=nn.Parameter(torch.zeros((self.memory_size,self.feature_shape[0],self.feature_shape[1]),dtype=torch.float32,requires_grad=False,device=device))
        self.optimizer = torch.optim.SGD([self.memory], lr=self.lr)
        if os.path.exists(self.memory_dir+"/loss_record.pt"):
            self.loss_record=torch.load(self.memory_dir+"/loss_record.pt").to(device)
        else:
            self.loss_record=torch.zeros(self.memory.shape,device=device)

if __name__=="__main__":
    memory=Memory(Load_dir="memory")
    print(memory.tot)
    memory.Save()
#
# author: Jingquan Lee
# date: 2020-06-03
# email: m13152143642@163.com
#

import os
import sys
import time
from tqdm import tqdm
import argparse

import torch
import numpy as np
from tensorboardX import SummaryWriter

from seir import model, encode, decode
from Dataset import Dataset

def train(args):
    train_dataset, validate_dataset = Dataset(args).split(args.num_for_train, args.num_for_validate)
    m = model([3, 36, 25], [5, 48], [25, 48, 25], 48)
    optimizer = torch.optim.Adam(m.parameters(), lr=args.rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m.to(device)
    print_validate = args.num_for_train // args.num_for_validate
    print("model:", m)
    with SummaryWriter(logdir=args.log_path) as writer:
        for e in range(args.epochs):
            print('%d epochs start at'%e, time.asctime(time.localtime(time.time())))
            print('learning rate: %f', args.rate)
            validate = iter(validate_dataset)
            pbar = tqdm(total=len(train_dataset)) if not args.without_process_bar else None
            print('train_dataset size:', len(train_dataset))
            print('validate_dataset size:', len(validate_dataset))
            for b, (density, mean_density, infection, A_regions) in enumerate(train_dataset):
                m.train()
                encode_density = density_normalize(density[0].to(device))
                encode_mean_density = mean_density_normalize(mean_density[0].to(device))
                encode_infection = infection_normalize(infection[0].to(device))
                encode_A_regions = A_regions[0].to(device)
                decode_density = density_normalize(density[1].to(device))
                decode_mean_density = mean_density_normalize(mean_density[1].to(device))
                decode_infection = infection_normalize(infection[1].to(device))
                decode_A_regions = A_regions[1].to(device)
                A = train_dataset.A.to(device)
                num_steps = encode_infection.shape[1]-1
                x = torch.cat([encode_density[:, 0], 
                        encode_mean_density[:, 0], 
                        encode_infection[:, 0]], dim=-1)
                label = [encode_density[:, 1:], encode_mean_density[:, 1:], encode_infection[:, 1:]]
                y, seir_feature = encode(m, A, encode_A_regions[:, 1:], 
                                x, encode_mean_density[:, 0],
                                label, num_steps)
                label = [decode_density, decode_mean_density, decode_infection]
                y, seir_feature = decode(m, A, decode_A_regions, 
                                seir_feature, y[1][:, -1],
                                density[1].shape[1])
                loss = m.compute_loss(y, label, weight=torch.Tensor([[.1],[.1],[.8]]).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar('scalar/loss', 
                                loss, global_step=e*len(train_dataset)+b)

                if b%print_validate == 0:
                    m.eval()
                    density, mean_density, infection, A_regions = next(validate)
                    encode_density = density_normalize(density[0].to(device))
                    encode_mean_density = mean_density_normalize(mean_density[0].to(device))
                    encode_infection = infection_normalize(infection[0].to(device))
                    encode_A_regions = A_regions[0].to(device)
                    decode_density = density_normalize(density[1].to(device))
                    decode_mean_density = mean_density_normalize(mean_density[1].to(device))
                    decode_infection = infection_normalize(infection[1].to(device))
                    decode_A_regions = A_regions[1].to(device)
                    A = validate_dataset.A.to(device)
                    x = torch.cat([encode_density[:, 0], 
                            encode_mean_density[:, 0], 
                            encode_infection[:, 0]], dim=-1)
                    label = [encode_density[:, 1:], 
                            encode_density[:, 1:], 
                            encode_infection[:, 1:]]
                    num_steps = encode_density.shape[1]-1
                    y, seir_feature = encode(m, A, encode_A_regions[:, 1:], 
                                    x, encode_mean_density[:, 0],
                                    label, num_steps)
                    y, seir_feature = decode(m, A, decode_A_regions,
                                    seir_feature, y[1][:, -1], decode_infection.shape[1])
                    error = m.evaluate(infection_recover(y[-1]),
                            infection[1].to(device))
                    writer.add_scalar('scalar/validate', error,
                            global_step=e*len(validate_dataset)+b//print_validate)
                if pbar:
                    pbar.update(1)
            if pbar:
                pbar.close()
            if e%args.save_per_epochs == 0:
                torch.save(m.state_dict(), os.path.join(args.save_path, "model_%d"%e))
            scheduler.step()

def normalize(x):
    return torch.log(x+1)

def recover(x):
    return torch.exp(x)-1

def infection_normalize(x):
    x = normalize(x)
    x = (x+5)/20
    return x

def infection_recover(x):
    x = x*20-5
    x = recover(x)
    return x

def density_normalize(x):
    x = normalize(x)
    x = x/20
    return x

def density_recover(x):
    x = x*20
    x = recover(x)
    return x

def mean_density_normalize(x):
    x = normalize(x)
    x = x/5
    return x

def mean_density_recover(x):
    x = 5*x
    x = recover(x)
    return x

def num_of_decode(e):
    return int(10*(1-np.cos(e/500*np.pi)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_for_train', type=int, default=9)
    parser.add_argument('--num_for_validate', type=int, default=1)
    parser.add_argument('--without_process_bar', action='store_true')
    #parser.add_argument('--total_num_steps', type=int, default=44)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--rate', type=float, default=3e-3)
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument('--save_path', type=str, default='./log')
    parser.add_argument('--batch_size', type=int, default=10)
    #parser.add_argument('--val_encode_steps', type=int, default=35)
    parser.add_argument('--density_file', type=str, default='../preprocess/loc_inf.csv')
    parser.add_argument('--infections_file', type=str, default='../preprocess/infection.csv')
    parser.add_argument('--seir_A_file', type=str, default='../preprocess/seir_A.csv')
    parser.add_argument('--migration_dir', type=str, default='../preprocess/')
    parser.add_argument('--save_per_epochs', type=int, default=10)
    args = parser.parse_args()
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    with open(os.path.join(args.log_path, 'configs.txt'), mode='w+') as f:
        f.write(str(args))
        f.flush()
        f.close()
    train(args)

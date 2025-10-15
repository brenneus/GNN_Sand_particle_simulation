import numpy as np
import torch

tensors = torch.load('./tensorFiles/position_tensor_test.pt')
print(len(tensors))
velos = []
accels = []
for traj, pos in enumerate(tensors):
    traj_velos = []
    traj_accels = []
    print(f"Trajectory {traj} shape: {pos.shape},")
    for t in range(1, len(pos)-1):
        velo = pos[t] - pos[t-1]
        accel = pos[t+1] - 2*pos[t] + pos[t-1]

        # print(pos[t].shape, velo.shape, accel.shape)
        traj_velos.append(velo)
        traj_accels.append(accel)
    torch_traj_velos = torch.stack(traj_velos)
    velos.append(torch_traj_velos)
    
    torch_traj_accels = torch.stack(traj_accels)
    accels.append(torch_traj_accels)

torch.save(velos, './tensorFiles/velocity_tensor_test.pt')
torch.save(accels, './tensorFiles/accel_tensor_test.pt')
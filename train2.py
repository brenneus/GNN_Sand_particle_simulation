import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from GNN import  GNN
import torch.optim as optim
import matplotlib.pyplot as plt
import random

#file imports
from build_graph import build_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_std = 0.00003 #standard deviation of the noise added to the velocities of the particles
delta = 2.5 #time step between particles

#wall boundary values
left_bottom_boundary = 0.1
top_right_boundary = 0.9

#for data loader to work, we need to either use a existing dataset from torch OR
#we need to create a custom dataset
class RandomTrajectoryAndTime(Dataset):
    def __init__(self, positions):
        self.positions = positions
    
    #this is used when the getitem is sampling(it samples within the # of trajectories)
    def __len__(self):
        return len(self.positions) #this should give the # trajectories
    
    #this method is what the Data Loader uses when sampling
    #idx is a randomely chosen trajectory
    def __getitem__(self, traj):
        #idx = randomely sampled trajectory 
        #get position, velocity, and accel tensors for chosen trajectory
        traj_positions = self.positions[traj]

        #we need to randomly sample a time step within the trajectory
        num_timesteps = traj_positions.shape[0] #this will give the accurate number = 320
        #ASK ABOUT THIS -- does the 5 prev velocities including the velocity between
        #   the current time step and next time step (4) or should the last velocity be 
        #   the one with the current time step and the prev (5)
        time_step = torch.randint(4, num_timesteps -2, (1,)).item() 

        return [traj, time_step]#traj_positions[time_step]

#------------------------------------------------------------------------------------
simulator = GNN(num_node_features=16, num_edge_features=3, num_message_passing_steps=10).to(device)

# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)  # Bias initialized to zero

# # Apply the weight initialization to the model
# simulator.apply(init_weights)


optimizer = optim.Adam(simulator.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss() #L2
simulator.train()
#---------------------------------------------------------------------------------------------
#Now, we use the Custom Dataset in the Dataloader + Train
def train(all_positions, all_velocities, all_accels):
    epochs = 40
    #creates an instance of the above custom class
    customDataset = RandomTrajectoryAndTime(all_positions)
    #print('done ')
    # first_data = customDataset[0]
    # traj, time = first_data
    # print((traj, time))

    #Creates a dataloader with a specified number of batches
    #num_worker = number of subprocesses using for data loading -- larger = speed up
    #trajs = [i for i in range(len(all_positions))]

    allEpochs = []
    for epoch in range(epochs):
        # batches = 32
        # batch = []
        # #train_loader = DataLoader(customDataset, batch_size=batches, shuffle=True)#, num_workers=2)
        # for i in range(batches):
        # train_loader = DataLoader(customDataset, batch_size=1, shuffle=True)
        samples = []
        #random_order = random.shuffle(trajs)
        for traj in range(len(all_positions)):
            for time_step in range(5, 319):                            #stays in bounds for prev 5 velocities
                samples.append((traj, time_step))
        
        random.shuffle(samples) 

            # print(type(train_loader))
            # batch.append(train_loader)

        
        # for i in train_loader:
        #     print(train_loader[i])
        #print(len(batch))

        # #loop through each of the batches
        # #training loop
        # # for epoch in range(epochs):  #when we want to introduce epochs
        # seen = set()
        all_loss = []
        runningSum = 0
        numPart = 0
        sumSquares = 0

        count = 0
        # for train_loader in batch:
        print("NEXT TRAIN LOADER---------------------------------------")
        for randTuple in samples:
            traj, time_step = randTuple
            # for traj, time_step in train_loader:
            #print((traj, time_step, count))
            count += 1
            #traj, time_step = train_loader[0], train_loader[1]
            # if temp in seen:
            #     continue
            # else:
            #     seen.add(temp)
            #print(traj.shape)
            # traj = traj.item()
            # time_step = time_step.item()
            positions_t = all_positions[traj][time_step]     #calculate positions at timestep
            #print("Positions, ", positions_t.shape)
            #TO DO: add noise to velocities...?
            #Change to [rand_time-5, rand_time] if time_step in method starts at 5
            #print(all_velocities[1][0])
            velocities_t = all_velocities[traj][time_step-4:time_step+1]
            # print("Velcoities:  ", velocities_t.shape)
            #print(type(all_velocities))

            # print(type(velocities_t))
            # print(type(position_t))
            try:
                flatter_velos = velocities_t.view(-1, 10)
            except:
                print("ERROR HERE-----------------------------")
                print(traj, time_step)
                print(velocities_t.shape)
                print("ERROR HERE-----------------------------")
                continue

            #TO DO (?) - "subtract the noise added to the most recent input velocity" - paper
            #   noise = touch.normal(0, self.noise_std, size=velocities_t.shape).to(velocity_t.device)
            #   acceleration_t = (traj_acceleration[rand_time] / self.delta_t) - noise
            acceleration_t = all_accels[traj][time_step].to(device)
            # print("Accelerations:  ", acceleration_t.shape[0])
            # print(acceleration_t.shape)
            optimizer.zero_grad() #clears previous gradients

            #calculate boundary conditions
            left_bottom = positions_t - left_bottom_boundary
            top_right = top_right_boundary - positions_t

            #Forward Pass + get predicted acceleration values
            graph = build_graph(positions_t, flatter_velos, left_bottom, top_right) # graph is on device
            # print(graph)
            # print(time_step, traj)
            predicted_acceleration = simulator(graph)
            # print("Predicted Accelerations: ", predicted_acceleration)
            runningSum += torch.abs(acceleration_t).sum()
            numPart += acceleration_t.shape[0]
            # sumSquares += temp.sum()
            
            #if the loss degrades or blows up too much
            if torch.any(torch.isnan(predicted_acceleration)) or torch.any(torch.isinf(predicted_acceleration)):
                print(f"nan or inf in predictions")
            
            mean = acceleration_t.mean(dim=0, keepdim=True)
            std = acceleration_t.std(dim=0, keepdim=True)
            # mean = runningSum / numPart
            # print(numPart, runningSum, mean)

            # sumSquares += ((acceleration_t - mean)**2).sum()
            # std = torch.sqrt(sumSquares / (numPart - 1))
            # Normalize the array to have 0 mean and unit variance
            normalized_acceleration_t = (acceleration_t - mean) / std
            # Calculate the loss
            loss = loss_func(predicted_acceleration, normalized_acceleration_t)
            # Backward pass: Compute gradients
            #print(type(loss))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(simulator.parameters(), 1.0)

            # Update model parameters
            optimizer.step()

            print(f'loss: {loss.item():.3f}')
            all_loss.append(loss.item())

        if epoch % 5 == 0:
            save_checkpoint(simulator, optimizer, epoch, loss, filename='model_checkpoint.pth')
        #     break
        # break
        tempAvg = sum(all_loss) / len(all_loss)
        allEpochs.append(tempAvg)
        
    return allEpochs


def save_checkpoint(model, optimizer, epoch, loss, filename='model_checkpoint.pth'):
    # Save the model's state_dict (parameters) and the optimizer state_dict (if needed)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename} for epoch {epoch}")

def loss_figure(all_loss, name):
    plt.figure(figsize=(10, 6))

    # Plot the loss values
    plt.plot(all_loss, label='Loss', color='b')

    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.ylim(0,2)
    # Optionally add a grid
    plt.grid(True)

    # Display the plot
    plt.savefig(name)

#MAIN
def main():
    #Load data
    all_positions = torch.load("./tensorFiles/position_tensor_test.pt")  #Replace with correct path to files
    all_velocities = torch.load("./tensorFiles/velocity_tensor_test.pt") #Replace with correct path to files
    all_accels = torch.load("./tensorFiles/accel_tensor_test.pt")        #Replace with correct path to files

    all_loss = train(all_positions, all_velocities, all_accels)

    loss_figure(all_loss, "Average Loss over 40 Epochs")

main()
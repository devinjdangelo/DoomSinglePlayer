import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools as it
from skimage import transform
    
def clean_human_action(action_list):
    cancel_action_pairs = [(0,1),(2,3),(4,5)]
    for action_pair in cancel_action_pairs:
        i,j = action_pair
        if action_list[i]==1 and action_list[j]==1:
            action_list[i] = 0
            action_list[j] = 0
            
    if sum(action_list[-2:])>1:
        #if the human player is holding "attack" and presses one of the switch weapon buttons
        #the switch weapon action overrides the attack action. The below assumes that in doom2.cfg
        #attack is listed before the switch weapon buttons
        #also assumed we cannot press two switch weapon buttons simultaneously. This is possible but 
        #hard to do accidentally. 
        action_list[-10:] = [0,1]

def get_indecies_from_list(action_list,buttons,actions):
    buttons = np.cumsum([0] + buttons)

    action_list = [int(action) for action in action_list]
    action_list_groups = [action_list[buttons[i]:buttons[i+1]] for i in range(len(buttons)-1)]
    indecies = [action.index(action_list_groups[i]) for i,action in enumerate(actions)]
        
    return indecies


def action_indecies_to_tensor(batch_indecies,num_offsets,num_measurements,num_actions):
  #batch_indecies is array of action indecies
  #which indicate which action of the 4 categories was chosen
  #num_actions indicates the total actions in stream to be converted
  #this helper function converts this array into a tensor of 0s and 1s
  #which can be multiplied with the action stream tensor outputs to
  #select the relevent action prediction to be compared with actual
  #action. Output shape must be (?,num_actions,num_offsets,num_measurements)

  #print(batch_indecies)  
  #print(num_actions)
    n_batches = len(batch_indecies)
    out_tensor = np.zeros(shape=(n_batches,num_actions,num_offsets,num_measurements),dtype=np.float32)
    for batch,action_chosen in enumerate(batch_indecies):
        out_tensor[batch,action_chosen,:,:] = 1
            
    return out_tensor


def compute_circles_visited(xcords,ycords,verbose=False):
     #this is the most important reward
     #it will reward the agent based on the total area
     #explored. this will be imputed using the xy cordinates
     #mindistance is in map units... 256 is approx 2x the width of a normal hallway
     mindistance = 128 
     #A = (mindistance/2)**2 * 3.14 #circles of radius 1/2 mindistance
     mindistance = mindistance**2 #distance is squared so we don't have to square root in distance formula
        
     coords = np.asarray(list(zip(xcords,ycords)))
     keepcoords = coords
     i = 0
     while i < keepcoords.shape[0]-1:
         refc = keepcoords[i]
         distance = np.sum((keepcoords - refc)**2,axis=1)
         keepcoords = np.asarray([j for n, j in enumerate(keepcoords) if distance[n] > mindistance or n<=i])
         i += 1    
     if verbose:
        print("Over ",coords.shape[0]," Tics, you traveled to ",len(keepcoords)," unique circles.")
     #area = A * len(keepcoords)
     #return area
     keepx = keepcoords[:,0].tolist()
     keepy = keepcoords[:,1].tolist()
     return len(keepcoords),keepx,keepy #number of unique circles should be sufficient info

        

doom2_monsters = ["Zombieman","ShotgunGuy","ChaingunGuy","WolfensteinSS","DoomImp","Demon","Spectre","LostSoul",
"BetaSkull","Cacodemon","BaronOfHell","HellKnight","Revenant","Arachnotron","Fatso","PainElemental",
"Archvile","Cyberdemon","SpiderMastermind","CommanderKeen","BossBrain","BossEye","BossTarget","SpawnShot","SpawnFire"]

#I measured empiracly the approximate maximum angle of aim between
#monster and player which would still usually score a hit
#So if player fires and angle <= max_angle_to_hit(distance)
#I will say that player successfully "fired at enemy"

distances = [5000,10000,20000,30000,60000,124160,150000,400000]
max_angles = [24,12,8,6,4,3,2,1]
max_angles = [1.5*theta for theta in max_angles]
max_angle_to_hit = lambda distance: np.interp(distance,distances,max_angles)


def detect_hits(labels,agent,melee=False):
  #detects if agent scored a hit with a hitscan weapon
  #assuming he fires successfully in the current state
  #each label in labels has label.object_position_x
  #y and z. Also label.object_name can be compared to
  #doom2_monsters to see if the object is an enemy.
  #agent[0] = xpos. agent[1]=ypos, agent[2]=angle

      monster_angles = []
      distances = []
      for label in labels:
          if label.object_name in doom2_monsters:
              x = label.object_position_x - agent[0]
              y = label.object_position_y - agent[1]
              angle = np.angle(x + 1j*y,deg=True)
              dist = x**2+y**2
              if angle<0:
                  angle += 360
                  
              monster_angles.append(angle)
              distances.append(dist)
      
      if melee:
          maxdist = 81**2 #can't hit regardless of angle if >81 units away
          monster_angles = [monster_angles[i] for i,dist in enumerate(distances) if dist<=maxdist]
          distances = [distances[i] for i,dist in enumerate(distances) if dist<=maxdist]
          
      if len(monster_angles)>0:
            #convert distances to angle tolerances (your aim can be more off when you are close)
         angle_tolerance = max_angle_to_hit(distances)
         hit_differences = np.absolute(monster_angles - agent[2])
         hits = [diff<angle_tolerance[i] for i,diff in enumerate(hit_differences)]
         any_hit = np.amax(hits)
      else:
         any_hit = False
    
      return any_hit


def get_targets(m,offsets):
    f = np.zeros([len(m),len(offsets),m.shape[1]])
    for i,offset in enumerate(offsets):
        f[:-offset,i,:] = m[offset:,:] - m[:-offset,:]
        if i > 0:
            f[-offset:,i,:] = f[-offset:,i-1,:]
    return f


def gen_random_mask(mask_len,out_len,max_gap,min_gap=1):
    assert mask_len/max_gap < out_len-1
    assert mask_len//out_len > min_gap+1
    num_zeros = mask_len-out_len
    zeros = np.zeros(num_zeros,dtype=np.bool)
    rolls = np.random.normal(loc=num_zeros/(out_len-1),scale=num_zeros/(2*out_len),size=out_len-2)
    rolls = np.clip(rolls.astype(int),min_gap+1,max_gap-1)
    missed_sum = num_zeros-np.sum(rolls)
    
    if missed_sum>max_gap:
        number_to_add = missed_sum - max_gap//2
        not_max = np.where(rolls<max_gap-2)
        while number_to_add>len(not_max[0]):
            rolls[not_max] += 1   
            number_to_add -= len(not_max[0])
            not_max = np.where(rolls<max_gap-2)
        indecies_to_increment = np.random.choice(not_max[0],size=number_to_add,replace=False)
        rolls[indecies_to_increment]+=1 
    elif missed_sum<0:
        number_to_add = max_gap//2 - missed_sum
        not_min = np.where(rolls>min_gap+2)
        while number_to_add>len(not_min[0]):
            rolls[not_min] -= 1   
            number_to_add -= len(not_min[0])
            not_min = np.where(rolls>min_gap+2)
            
        indecies_to_decrement = np.random.choice(not_min[0],size=number_to_add,replace=False)
        rolls[indecies_to_decrement]-=1
        
    break_points = np.cumsum(rolls)
    gaps = np.split(zeros,break_points)
    mask = [[True]] * (2*out_len-1)
    mask[1:-1:2] = gaps
    mask = np.concatenate(mask)
    return mask.tolist()
    
def find_longest_gap(mask,check_for):
    longest = 0
    shortest = len(mask)
    current = 0
    for num in mask:
        if num == check_for:
            current += 1
        else:
            longest = max(longest, current)
            shortest = min(shortest,current)
            current = 0

    return max(longest, current), min(shortest,current)
  
    
def merge_batches(batch1,batch2):
    if len(batch1)==5 and len(batch2)==5:
        frames = np.concatenate([batch1[0],batch2[0]])
        m = np.concatenate([batch1[1],batch2[1]])
        ahist = np.concatenate([batch1[2],batch2[2]])
        ataken = np.concatenate([batch1[3],batch2[3]])
        target = np.concatenate([batch1[4],batch2[4]])
        return frames,m,ahist,ataken,target
    elif len(batch1)==6 and len(batch2)==6:
        frames = np.concatenate([batch1[0],batch2[0]])
        m = np.concatenate([batch1[1],batch2[1]])
        ahist = np.concatenate([batch1[2],batch2[2]])
        ataken = np.concatenate([batch1[3],batch2[3]])
        target = np.concatenate([batch1[4],batch2[4]])
        lbuffer = np.concatenate([batch1[5],batch2[5]])
        return frames,m,ahist,ataken,target,lbuffer
    else:
        raise ValueError('batches are not both of length 5 or 6')
        
def stream_to_PV(stream,drate):
    t = stream.shape[1]
    e = stream.shape[0]
    def PV(i):
        dr = ([1]*i) + [(drate**j) for j in range(t-i)]
        dr = np.tile(dr,e).reshape(e,t)
        return np.sum(np.multiply(stream,dr),axis=1)
    PVs = [PV(i) for i in range(t)]
    PVs = np.stack(PVs,axis=1)
    return PVs
    
    


def get_pc_target(sbuff,local_means=False):
    #calculate pixel control targets from most recent 2 frames
    #sbuff 100x160x3x2 CEILAB images (stored as 100x160x6)
    #output is 25x25
    
    cropped_diff = np.absolute(sbuff[2:-2,2:-2,3:] - sbuff[2:-2,2:-2,:3])
    #numpy and skimage implementation which are slightly different... not sure why
    if local_means:
        targets = transform.downscale_local_mean(cropped_diff,[4,4,3])[:,:,0]
    else:
        cropped_diff = np.reshape(cropped_diff,[4,20,4,20,3])
        targets = np.mean(cropped_diff,axis=(0,2,4))
        
    return targets

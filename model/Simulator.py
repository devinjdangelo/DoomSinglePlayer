
#import horovod.tensorflow as hvd
#hvd.init(comm=[[0,8],[1,2,3,4,5,6,7,9,10,11]])


from vizdoom import *
from oblige import *
import skimage as skimage
from skimage import color, transform
import itertools as it
import numpy as np
import csv
import imageio
import os
import contextlib
import random
import time

from model.Agent import DoomAgent
from model.Experience import ExperienceRecorder
from model.ExperienceH5 import H5ExperienceRecorder

from model.utils import *

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = comm.Get_size()

class DoomSimulator:
    def __init__(self,args):
        #valid modes:
        #record -- human player generates training data by playing Doom
        #pretrain -- agent learns from human data and plays test levels regularly to track performance gains
        #train -- agent begins exploring and learning from its own data, plays tests regularly to track performance

        color = 0 if rank<=7 else 1
        self.local_comm = MPI.COMM_WORLD.Split(color,rank)
        self.local_workers = self.local_comm.Get_size()

        if 'mode' not in args or 'framedims' not in args:
            raise ValueError('Mode and framedims are required arguments for DoomSimulator. \
                             Please add to args.')
            
        self.mode = args['mode']
        self._define_action_groups(args['num_action_splits'],args['a_size'],args['group_cond'])
        self.num_buttons = args['num_buttons']
        
        self.xdim,self.ydim = args['framedims']
        self.frame_skip = args['frame_skip']         
        self.test_stat_file = args['test_stat_file'] if 'test_stat_file' in args else None
        reset_file = args['reset_file'] if 'reset_file' in args else False
        self.model_path = args['model_path'] if 'model_path' in args else None
        self.gif_path = args['gif_path'] if 'gif_path' in args else ''
        
        self.num_measurements = args['num_measurements']
        
        if self.test_stat_file is not None and reset_file:
            with open(self.test_stat_file, 'w') as teststatfile:
                    wr = csv.writer(teststatfile)
                    wr.writerow(['Episode','Kills','Explored','Wins','Total Reward','Kills Avg','Explore Avg','Wins Avg','Rewards Avg'])
                    
        self.env = DoomGame()
        self.doompath = args['doom_files_path'] 
        self.env.load_config(self.doompath+'doom2.cfg')
        self.env.set_button_max_value(TURN_LEFT_RIGHT_DELTA,10)
        doomengine = args['doom_engine']
        self.env.set_doom_game_path(self.doompath+doomengine)
        self.env.set_labels_buffer_enabled(True)
        if self.mode=='record':
            self.env.set_window_visible(True)
            self.env.set_mode(Mode.SPECTATOR)
        elif self.mode=='pretrain' or self.mode=='train':
            args['group_buttons'] = self.group_buttons
            args['group_actions'] = self.group_actions
            self.env.set_mode(Mode.PLAYER)
            if self.model_path is None:
                raise ValueError('If mode is pretrain or train, must provide filepath for Agent network.')
            self.agent = DoomAgent(args)
            with open('model_stats.csv', 'w') as f:
                wr = csv.writer(f)
                wr.writerow(['iter','Policy Loss','Critic Loss','Entropy','Grad Norm'])
        else:
            raise ValueError('DoomSimulator has no mode, ',str(self.mode),'.')
        
        if self.mode=='record' or self.mode=='pretrain':
            if 'mem_location' not in args:
                raise ValueError('If mode is recording or pretrain, must provide filepath for ExperienceRecorder. \
                                 use key "mem_location')
            self.experience_memory = H5ExperienceRecorder(args)
        else:
            if args['use_human_data']:
                self.using_human_data = True
                self.human_data = H5ExperienceRecorder(args)
                self.hsize = args['hsize']
            else: 
                self.using_human_data = False
                self.hsize = 0
            self.experience_memory = ExperienceRecorder(args)
            
        self.generator = DoomLevelGenerator()
        self.num_maps=0
        self.map=0

        self.train_epochs = args['epochs_per_policy']
        
        #these linearly annealed over each set of epochs_per_policy
        self.lr_schedule = args['learning_rate']
        self.clip_schedule = args['clip_e']
        
        self.episode_timeout_steps = args['episode_length']            
        
    def _define_action_groups(self,splits,a_sizes,group_cond):
         
        self.group_actions = []
        for i in range(splits):
            group_i = [list(a) for a in it.product([0, 1], repeat=a_sizes[i])]
            group_i = [a for a in group_i if group_cond[i](a)]
            self.group_actions.append(group_i)
            
        self.group_buttons = a_sizes
    def start_new_level(self,won_level=False):
        if self.num_maps==0 and not won_level:
            self.env.close()
            #size = random.choice(['micro','tiny','small','regular'])
            size = random.choice(['micro','tiny'])
            theme = random.choice(["original", "mostly_original", "epi", "mostly_epi", "bit_mixed", "jumble", "tech", "mostly_tech", "urban", "mostly_urban", "hell", "mostly_hell"])
            self.generator.set_seed(random.randint(1,999))
            self.generator.set_config({"size": size,"theme":theme, "length": "episode","health": "normal",
                                       "weapons": "sooner","mons":"some"})
            wad_path = self.doompath + str(rank) + 'test.wad'
            with contextlib.suppress(FileNotFoundError):
                os.remove(wad_path)
                os.remove(self.doompath + str(rank) +  'test.txt')
                os.remove(self.doompath + str(rank) + 'test.old')
            print('Generating new maps...')
            self.num_maps = self.generator.generate(wad_path,verbose=False)
            self.last_map = self.num_maps
            print('Done!')
            #this should work during runtime now
            self.env.set_doom_scenario_path(wad_path)
            self.env.init()
            self.map = 0
            self.next_map = 1
        if won_level:
            if self.map==self.last_map:
                self.map = 1
            else:
                self.map += 1
        else:
            self.map = self.next_map
            self.next_map += 1
            self.num_maps -= 1
        strmap = "map{:02}".format(self.map)
        self.env.set_doom_map(strmap)
        self.env.new_episode()
        
    def restart_level(self):
        self.env.new_episode()
        
    def process_game_vars(self,m_raw,labels):
        
        self.selected_weapon = m_raw[1]
        
        fist_active = 1 if self.selected_weapon==1 else 0
        pistol_active = 1 if self.selected_weapon==2 else 0
        shotgun_active = 1 if self.selected_weapon==3 else 0
        chaingun_active = 1 if self.selected_weapon==4 else 0
        rocket_active = 1 if self.selected_weapon==5 else 0
        plasma_active = 1 if self.selected_weapon==6 else 0
        bfg_active = 1 if self.selected_weapon==7 else 0
        super_active = 1 if self.selected_weapon==8 else 0
        chainsaw_active = 1 if self.selected_weapon==9 else 0

        #weap1 = 0 if fist only and =1 if fist and chainsaw
        has_fist = 1 if m_raw[10]>0 else 0
        #weap2 = 1 if pistol
        has_pistol = 1 if m_raw[11]>0 else 0
        #weap3 = 1 if shotty 
        has_shotgun = 1 if m_raw[12]>0 else 0
        #weap4 = 1 if supershotty
        has_chaingun = 1 if m_raw[13]>0 else 0
        has_rocket = 1 if m_raw[14]>0 else 0
        has_plasma = 1 if m_raw[15]>0 else 0
        has_bfg = 1 if m_raw[16]>0 else 0
        has_super = 1 if m_raw[17]>0 else 0
        has_chainsaw = 1 if m_raw[18]>0 else 0
        
        #ammo2 = pistol bullets
        ammo2 = m_raw[20]
        #ammo3 = shotgun shells
        ammo3 = m_raw[21]
        #ammo4 = rockets
        ammo4 = m_raw[22]
        #ammo5 = cells
        ammo5 = m_raw[23]

        health = m_raw[2]
        armor = m_raw[3]

        
        self.monster_deaths.append(m_raw[5])
       
        current_x = m_raw[6]
        current_y = m_raw[7]
        self.level_ypos.append(current_x)
        self.level_xpos.append(current_y)
        if len(self.level_xpos) > 1:

            #make compute_circles_visited return xpos,ypos list of unique points to overwrite running appended list of all points
            area_explored,self.level_xpos,self.level_ypos = compute_circles_visited(self.level_xpos,self.level_ypos,verbose=False)
            self.level_explored.append(area_explored)
            
            if self.mode=='record' and self.level_explored[-1]>self.level_explored[-2]:
                print('You explored ',self.level_explored[-1]-self.level_explored[-2],' circles!')

            #labels has info about visible objects including enemies (used for hit detection)
            agent = [current_x,current_y,m_raw[29]]
            using_melee = True if fist_active or chainsaw_active else False
            hit_scored = detect_hits(labels,agent,melee=using_melee)

            if hit_scored and self.currently_attacking:
                #if aiming close to visible enemy and attack action in progress we score a "hit"
                self.hits += 1
                self.last_hit_n_ago = 0
                
            if self.last_hit_n_ago<=8:
                #if within 3 steps we scored a "hit" and an enemy dies we score a "kill"
                #need to check if 3 is enough for non hitscan weapons
                self.last_hit_n_ago+=1
                current_kills = max(self.monster_deaths[-1] - self.monster_deaths[-2],0)
                self.episode_kills += current_kills 
                #if current_kills!=0:
                    #print('You scored, ',current_kills,' kills!')
            

        else: 
            self.level_explored.append(0)
            area_explored = 0
            
        area_explored += self.previous_level_explored
        
        m = [fist_active,pistol_active,shotgun_active,chaingun_active,rocket_active,
             plasma_active,bfg_active,super_active,chainsaw_active,has_fist,has_pistol,has_shotgun,
             has_chaingun,has_rocket,has_plasma,has_bfg,has_super,has_chainsaw,
             ammo2,ammo3,ammo4,ammo5,health,armor,self.episode_kills,self.hits,area_explored,
             self.deaths,self.levels_beat]
   
        return m     
        
    def play_episode(self,training_step=0,testing=False,get_frames=False):
        #episode may consist of multiple levels
        #since vizdoom ends an episode when the level is completed,
        #we must catch when the episode ends but the player didn't die 
        #or timeout and continue the epsiode on the next level
        frames = []
        episode_steps = 0
        level_steps = 0
        episode_finished = False
        self.monster_deaths = []
        self.hits = 0
        self.episode_kills = 0
        self.previous_level_explored = 0
        self.level_explored = []
        self.level_ypos = []
        self.level_xpos = []
        self.currently_attacking = False
        self.deaths = 0
        self.levels_beat = 0
        episode_buffer = []
        self.last_hit_n_ago=16
        
        self.start_new_level()
        state = self.env.get_state()
        s = state.screen_buffer
        hud_px_height = self.ydim // 5 + 1 #HUD is 1/5th of the frame up from the bottom
        s = s[:-hud_px_height,:,:] #crop out the HUD
        s = skimage.transform.resize(s,(self.xdim,self.ydim,3))
        if self.colorspace == 'LAB':
            s = skimage.color.rgb2lab(s)
        elif self.colorspace == 'RGB':
            s = img_as_ubyte(s)
            s = s.astype(np.float32,copy=False)
        else:
            raise ValueError('Color space, ',self.colorspace,' is not defined.')

        abuffer = np.zeros(self.num_buttons,dtype=np.int32)
        m = self.process_game_vars(state.game_variables,state.labels)

        while not episode_finished:                
            if self.mode == 'record':
                attack_actions = []
                for _ in range(self.frame_skip):
                    self.env.advance_action()
                    #x,y,z,theta,attack,switch,use
                    action = self.env.get_last_action()
                    attack_actions.append(action[9])
                clean_human_action(action)
                aidx = get_indecies_from_list(action,self.group_buttons,self.group_actions)
                self.currently_attacking = max(attack_actions)
                episode_buffer.append([s,m,abuffer,aidx,0,0])
            elif self.mode == 'train' or self.mode=='pretrain':
                vz_action,aidx,a_prob,state_value = self.agent.choose_action(s,m,abuffer,training_step,testing,self.selected_weapon)
                self.env.make_action(vz_action.tolist(),self.frame_skip)    
                action = vz_action
                # state, measurements, previous action, prob of chosen action, value of current state
                episode_buffer.append([s,m,abuffer,aidx,a_prob,state_value])
                self.currently_attacking = True if vz_action[9] else False
                
            if self.env.is_episode_finished() and not self.env.is_player_dead():
                #catch if Vizdoom ended the episode because player completed level 
                print('you beat the level!')
                self.start_new_level(won_level=True)
                self.monster_deaths = []
                self.level_ypos = []
                self.level_xpos = []
                self.previous_level_explored += self.level_explored[-1]
                level_steps = 0
                self.levels_beat += 1
            elif self.env.is_player_dead():
                #if player dies restart level
                #do not reinitialize any variable!
                #no reward for exploring same part of map after respawning... 
                #kill rewards still count 
                #print('you died! restarting...')
                self.restart_level()
                self.deaths += 1
                #self.monster_deaths[-2:] = [0,0]
            
            if not episode_finished:
                level_steps+=1
                episode_steps+=1
                state = self.env.get_state()
                m_raw = state.game_variables
                m = self.process_game_vars(m_raw,state.labels)          
                
                s = state.screen_buffer
                s = s[:-hud_px_height,:,:] #crop out the HUD
                s = skimage.transform.resize(s,(self.xdim,self.ydim,3))
                if get_frames:
                    frames.append(s*255)
                if self.colorspace == 'LAB':
                    s = skimage.color.rgb2lab(s)
                elif self.colorspace == 'RGB':
                    s = img_as_ubyte(s)
                    s = s.astype(np.float32,copy=False)
                else:
                    raise ValueError('Color space, ',self.colorspace,' is not defined.')
                                    
                abuffer = action
                
                if episode_steps > self.episode_timeout_steps:
                    #print('timeout!')
                    episode_finished = True
                        
        
        self.previous_level_explored += self.level_explored[-1]                 
        return episode_buffer,self.episode_kills,self.previous_level_explored,episode_steps,frames,self.levels_beat

    def record_episodes(self):
        response = input('Ready to start playing? Enter (y) when ready or (n) to abort: ')
        episode=0
        while response != 'n':
            self.env.init()
            episode_buffer,kills,explored,steps,_ = self.play_episode()
            print('You scored, ',kills,' kills and explored ',explored,' units over ',steps,' steps or ',steps*4//35,' seconds')
            self.env.close()
            r = self.experience_memory.append_episode(episode_buffer)
            print('reward: ', r)
            print(self.experience_memory.n_episodes, ' have been recorded.')
            response = input('Starting new episode... Enter (y) when ready or (n) to abort: ')
            episode += 1
            
            
    def pre_train(self,size,test_every_n_iters=100):
        steps = 0
        i=0
        episode=0
        t=time.time()
        msteps_per_day = 0
        while True:
            batch = self.experience_memory.get_batch(size)
            loss,sharpness, g_n = self.agent.update_c(size,batch,steps)
            print('i: ',i,' loss: ',loss,' g_n: ',g_n,' Steps per Day ',msteps_per_day,' million')
            savelist = [i,loss,sharpness,g_n]
            with open('model_stats.csv', 'a') as f:
                wr = csv.writer(f)
                wr.writerow(['{:.2f}'.format(x) for x in savelist]) 
            steps += size*256
            if i % test_every_n_iters == 0:
                episode+=1
                self.agent.reset_state()
                self.agent.save(episode)
                self.env.init()
                _,kills,explored,_,frames = self.play_episode(training_step=steps,testing=True,get_frames=True)
                print('Agent scored, ',kills,' kills and explored ',explored,' units over ',steps,' steps or ',steps*4//35,' seconds')
                self.env.close()
                if self.test_stat_file is not None:
                   savelist = [episode,kills,explored]
                   with open(self.test_stat_file, 'a') as teststatfile:
                       wr = csv.writer(teststatfile)
                       wr.writerow(['{:.2f}'.format(x) for x in savelist])
                       
                frames = np.array(frames)
                frames = (frames*255).astype(np.uint8)        
                imageio.mimwrite(self.gif_path+'/testepisode'+str(episode)+'.gif',frames,duration=self.frame_skip/35)

            i +=1
            steps_per_sec = (steps)//(time.time()-t)
            msteps_per_day = 60*60*24*steps_per_sec/1000000
            #print('Steps per second',steps_per_sec,' Steps per day ',msteps_per_day,' million')
                
    def train(self,size,update_n,start_step=0,save_n=25,test_n=25,init_explore=None,init_reward=None,
                    init_win=None,init_kills=None):
        self.env.init()
        training_steps = start_step
        episode = 0
        t = time.time()
        explore_avg = init_explore if init_explore is not None else 20
        reward_avg = init_reward if init_reward is not None else 2
        wins_avg = init_win if init_win is not None else 0
        kills_avg = init_kills if init_kills is not None else 0.3
        while True:
            episode += 1
            save = episode % save_n == 0
            test = False
            episode_buffer,kills,explored,steps,frames,wins = self.play_episode(training_step=training_steps,testing=test,get_frames=save)
            self.agent.reset_state()
            rewards = self.experience_memory.append_episode(episode_buffer)
            print('Episode ',episode,' Agent scored, ',kills,' kills and explored ',explored,' units over ',steps,' steps or ',steps*4//35,' seconds')
            rewards,kills,explored,wins = self.sync_performance_stats(rewards,kills,explored,wins)

            if self.test_stat_file is not None:
                a_weight = max(1/(episode+1),1/10000*self.local_workers) if init_explore is None else 1/10000*self.local_workers
                explore_avg = (1-a_weight) * explore_avg + a_weight*explored
                training_steps += steps
                reward_avg = (1-a_weight) * reward_avg + a_weight*rewards
                wins_avg = (1-a_weight) * wins_avg + a_weight*wins
                kills_avg = (1-a_weight) * kills_avg + a_weight*kills
                
                savelist = [episode,kills,explored,wins,rewards,kills_avg,explore_avg,wins_avg,reward_avg]
                with open(self.test_stat_file, 'a') as teststatfile:
                    wr = csv.writer(teststatfile)
                    wr.writerow(['{:.4f}'.format(x) for x in savelist])
                           
            comm.Barrier()
            if episode % (update_n//n_workers)==0 and self.experience_memory.n_episodes>=size and (rank==0 or rank==8):
                iterations = (self.train_epochs * update_n//n_workers*self.local_workers) // size
                for i in range(1,int(iterations)+1):
                    f = 1 - (i-1)/iterations
                    lr = self.lr_schedule(f)
                    clip = self.clip_schedule(f)
                    batch = self.get_batch(size)
                    ploss,closs,entropy,g_n=self.agent.update(size,batch,training_steps,lr,clip)
                    print('i: ',episode,'policy score: ',ploss,' critic loss ',closs,' entropy ', entropy,' g_n: ',g_n) 

                    savelist = [episode,ploss,closs,entropy,g_n]
                    with open('model_stats.csv', 'a') as f:
                        wr = csv.writer(f)
                        wr.writerow(['{:.6f}'.format(x) for x in savelist])                  


            if episode % save_n ==0:
                if rank==0:
                    self.agent.save(episode)
                if rank==0 or rank==8:
                    frames = np.array(frames)
                    frames = (frames*255).astype(np.uint8)
                    imageio.mimwrite(self.gif_path+'/testepisode'+str(rank)+'r'+str(episode)+'.gif',frames,duration=self.frame_skip/35)            

            if rank==0:    
                steps_per_sec = (training_steps*n_workers-start_step)//(time.time()-t)
                msteps_per_day = 60*60*24*steps_per_sec/1000000
                print('Steps per second',steps_per_sec,' Steps per day ',msteps_per_day,' million')
            comm.Barrier()


    def get_batch(self,size):
        return self.experience_memory.get_batch(size)
    
    def sync_performance_stats(self,rewards,kills,explored,wins):
        rewards = np.array(rewards,dtype=np.float64)
        kills = np.array(kills,dtype=np.float64)
        explored = np.array(explored,dtype=np.float64)
        wins = np.array(wins,dtype=np.float64)
        sum_rewards = None
        sum_kills = None
        sum_explored = None
        sum_wins = None 
        if rank == 0 or rank==8:
            sum_rewards = np.empty((),dtype=np.float64)
            sum_kills = np.empty((),dtype=np.float64)
            sum_explored = np.empty((),dtype=np.float64)
            sum_wins = np.empty((),dtype=np.float64) 

        self.local_comm.Reduce([rewards,MPI.DOUBLE],[sum_rewards,MPI.DOUBLE],op=MPI.SUM,root=0)
        self.local_comm.Reduce([kills,MPI.DOUBLE],[sum_kills,MPI.DOUBLE],op=MPI.SUM,root=0)
        self.local_comm.Reduce([explored,MPI.DOUBLE],[sum_explored,MPI.DOUBLE],op=MPI.SUM,root=0)
        self.local_comm.Reduce([wins,MPI.DOUBLE],[sum_wins,MPI.DOUBLE],op=MPI.SUM,root=0)

        if rank==0 or rank==8:
            sum_rewards /= self.local_workers
            sum_kills /= self.local_workers
            sum_explored /= self.local_workers
            sum_wins /= self.local_workers

        return sum_rewards,sum_kills,sum_explored,sum_wins
        
                                   
            
            
            
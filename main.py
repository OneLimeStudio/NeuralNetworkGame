import pygame
import random
import neat
import os
import time
import pickle
SCREEN_WIDTH = 315
SCREEN_HEIGHT = 700
PLAYER = 75
TRAIN_WIDTH = 75
TRAIN_HEIGHT = 200
class train:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def _draw_(self,win):
        pygame.draw.rect(win,(255,255,255),(self.x,self.y,TRAIN_WIDTH,TRAIN_HEIGHT))
        self.y += 10



def Spawner():
    x = random.sample([15,15+15+15+TRAIN_WIDTH,2*(15+15+7.5+TRAIN_WIDTH)],2)
    return x
        
            

class player:
    
    def __init__(self , x,y):
        self.x  = x
        self.y   = y
        self.WIDTH = PLAYER
    def _draw_(self,win):
        pygame.draw.rect(win,(255,0,0),(self.x,self.y,self.WIDTH,self.WIDTH))
    
    def move(self,left,right):
        if left:
            self.x -= 15+15+PLAYER
        if right:
            self.x += 15+15+PLAYER
            
            
            
            
    def isAlive(self):
        pass

           
class Subway:    
    def __init__(self):
        self.z = Spawner()
        self.train1 = train(self.z[0], -30)
        self.train2 = train(self.z[1],-30)
        # flag = 0
        self.clock = pygame.time.Clock()
        pygame.init()

        self.window = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
        x = random.sample([15,15+15+15+PLAYER,2*(15+15+7.5+PLAYER)],1)
        self.Player = player(x[0],SCREEN_HEIGHT-PLAYER-70)
    def testai(self,net):
        self.clock.tick(60)
        run = True
        while run:
            self.clock.tick(60)
            self.window.fill((0,0,0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            #output = net.activate((abs(self.Player.x-self.train1.x),abs(self.Player.x - self.train2.x),abs(self.Player.y - self.train1.y),abs(self.Player.y - self.train2.y)))
            decision = output.index(max(output))
            if decision == 1:
                self.Player.move(True,False)
            elif decision == 2:
                self.Player.move(False ,True)
                
                
            if(self.train1.y > SCREEN_HEIGHT+20):
                self.z = Spawner()
                self.train1 = train(self.z[0],-30)
                self.train2 = train(self.z[1],-30)
            
            self.Player._draw_(self.window)
            self.train1._draw_(self.window)
            self.train2._draw_(self.window)
            pygame.display.update()
                
            
        
    def train_ai(self,genome,config):
        run = True
        start_time = time.time()
        net = neat.nn.FeedForwardNetwork.create(genome,config)
        self.genome = genome
        
        while run:
            self.clock.tick(60)
            self.window.fill((0,0,0))
            pygame.draw.line(self.window,(0,255,0),(SCREEN_WIDTH/3,0 ),(SCREEN_WIDTH/3, SCREEN_HEIGHT))
            pygame.draw.line(self.window,(0,255,0),(2*SCREEN_WIDTH/3,0 ),(2*SCREEN_WIDTH/3, SCREEN_HEIGHT))
            
            # if(trains[flag] > SCREEN_HEIGHT/2):
            #     trains[flag+1] = train(Spawner(), 0)
                
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return True
                    run = False
                    quit()
            
                # elif event.type == pygame.KEYDOWN:
                #     if event.key == pygame.K_LEFT:
                #         self.Player.move(True,False)
                #     elif event.key == pygame.K_RIGHT:
                #         self.Player.move(False,True)
            if(self.train1.y > SCREEN_HEIGHT+20):
                self.z = Spawner()
                self.train1 = train(self.z[0],-30)
                self.train2 = train(self.z[1],-30)
            
            self.movePlayer(net)
            self.train1._draw_(self.window)
            self.train2._draw_(self.window)
        
            self.Player._draw_(self.window)
            duration = time.time()- start_time
            
            pygame.display.update()
            if((self.Player.x == self.train1.x and self.train1.y+TRAIN_HEIGHT >= self.Player.y)or (self.Player.x == self.train2.x and self.train2.y+TRAIN_HEIGHT >= self.Player.y)or self.Player.x >SCREEN_WIDTH or self.Player.x < 0):
                print(duration)
                self.calculate_fitness(duration)
                break
        return False
        
                
    def movePlayer(self,net):
        #output = net.activate((abs(self.Player.x-self.train1.x),abs(self.Player.x - self.train2.x),abs(self.Player.y - self.train1.y),abs(self.Player.y - self.train2.y)))
        output = net.activate((abs(self.Player.x-self.train1.x),abs(self.Player.x-self.train2.x),abs(self.Player.y - self.train1.y + TRAIN_HEIGHT),abs(self.Player.y - self.train2.y+TRAIN_HEIGHT)))
        decision = output.index(max(output))
        if(decision!= 0):
            print(decision,"LALALALLA")
        if (self.Player.x == 15+15+15+PLAYER):
            self.genome.fitness += 0.03

        elif decision == 0:
            self.Player.move(True,False)
            self.genome.fitness += 0.05
        elif decision == 1:
            self.Player.move(False,True)
            self.genome.fitness += 0.05
        if(self.Player.x >SCREEN_WIDTH or self.Player.x < 0):
            self.genome.fitness -= 1
    def calculate_fitness(self,duration):
        self.genome.fitness += duration
        
def eval_genomes(genomes,config):
    for i ,(genome_id1,genom1) in enumerate(genomes):
        #print(round(i/len(genomes) * 100), end=" ")
        genom1.fitness = 0
        game = Subway()
        force_Quit = game.train_ai(genom1,config)
        if force_Quit:
            quit()
        
def run_neat(config):
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-69')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    winner = p.run(eval_genomes, 100)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
def test_best_network(config):
    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    game = Subway()
    game.testai(winner_net)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    configPath = os.path.join(local_dir,'config.txt')
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,neat.DefaultSpeciesSet, neat.DefaultStagnation,configPath)
    run_neat(config)

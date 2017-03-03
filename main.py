#!/usr/bin/env python3

import gym
import tensorflow as tf
import numpy
import random
import collections
import shutil

# Game
ENV_NAME = "CartPole-v0"

# NN
DESCENT_RATE = 0.0001
HIDDEN_LAYER_NODES = [5] # There are at most 5 peaks on the histogram

# RL
INIT_STDDEV = 0.001
INIT_MEAN = 0
INITIAL_EPS = 0.5
FINAL_EPS = 0.01
GAMMA = 0.9
EPISODE = 10000
STEP = 300
REPLAY_SIZE = 1000
BATCH_SIZE = 32

# Test
TEST_INTERVAL = 100
TEST_CASES = 3

# Summary
SUMMARY_INTERVAL = 100

class Trainer:
    ''' Receive transissions to train '''

    def __init__(self, inLayer, outLayer):
        ''' Add nodes to the net to calculate loss
            @param inLayer : input layer in the net
            @param outLayer : output layer in the net '''
        with tf.name_scope('loss'):
            self.sampledState = inLayer
            self.sampledAction = tf.placeholder(tf.float32, outLayer.get_shape())
            curQ = tf.reduce_sum(self.sampledAction * outLayer, 1) # filter out Q value for selected action
            self.sampledQ = tf.placeholder(tf.float32, curQ.get_shape())
            diffs = self.sampledQ - curQ
            loss = tf.reduce_mean(tf.square(diffs))
            self.optimizer = tf.train.AdamOptimizer(DESCENT_RATE).minimize(loss)

            tf.summary.histogram('diffs', diffs)
            tf.summary.scalar('loss', loss)

    def getFeedDict(self, sampledState, sampledAction, sampledQ):
        ''' Train the net with samples
            @param sess : tf session
            @param sampledState : [sampleId][stateId] = whether this sample is of this state
            @param sampledAction : [sampleId][actionId] = whether this sample is of this action
            @param sampledQ : [sampleId] = sampled Q value in this sample '''

        return {
            self.sampledState: sampledState,
            self.sampledAction: sampledAction,
            self.sampledQ: sampledQ
        }

class Net:
    ''' A MLP '''

    def __init__(self, dims):
        ''' Constructor
            @param dims : dimensions of each layer. '''

        self.dims = dims
        self.layers = [] # [0] = input, [-1] = output
        self.layers.append(tf.placeholder(tf.float32, (None, dims[0])))
        for i in range(1, len(dims)):
            with tf.name_scope('layer' + str(i)):
                coe = self._randomVariable('coe', (dims[i - 1], dims[i]))
                bias = self._randomVariable('bias', (dims[i], )) # tf supports Broadcasting, so it's of 1D. And here is an extra comma
                self.layers.append(tf.nn.relu(tf.matmul(self.layers[-1], coe) + bias))
        self.trainer = Trainer(self.layers[0], self.layers[-1])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        shutil.rmtree('logs')
        self.summaryWriter = tf.summary.FileWriter('logs', self.sess.graph)
        self.summaries = tf.summary.merge_all()
        self.trainCnt = 0

    def getQs(self, state):
        ''' Get Q values of state `state` '''

        return self.sess.run(self.layers[-1], {
            self.layers[0]: (state, ) # yes, here is an extra comma
        })[0]

    def feed(self, sampledState, sampledAction, sampledQ):
        ''' Train '''

        toMat = lambda sample, length: list(map(lambda x: self._oneHotKey(x, length), sample))
        feedDict = self.trainer.getFeedDict(sampledState, toMat(sampledAction, self.dims[-1]), sampledQ)
        if self.trainCnt % SUMMARY_INTERVAL == 0:
            summary, _ = self.sess.run((self.summaries, self.trainer.optimizer), feedDict)
            self.summaryWriter.add_summary(summary, self.trainCnt)
        else:
            self.sess.run(self.trainer.optimizer, feedDict)
        self.trainCnt += 1

    @classmethod
    def _randomVariable(cls, name, shape):
        ''' Generate a variable node with random initial values of shape `shape` '''

        with tf.name_scope(name):
            ret = tf.Variable(tf.truncated_normal(shape, mean = INIT_MEAN, stddev = INIT_STDDEV))
            tf.summary.histogram('histogram', ret)
            return ret

    @classmethod
    def _oneHotKey(cls, x, length):
        ''' x => [x] = 1 '''

        return [1 if i == x else 0 for i in range(length)]

class Agent:

    def __init__(self, env):
        self.env = env
        self.net = Net([env.observation_space.shape[0]] + HIDDEN_LAYER_NODES + [env.action_space.n])
        self.eps = INITIAL_EPS
        self.replay = collections.deque()

    def reset(self):
        ''' Reset to the initial state '''

        self.state = self.env.reset()

    def optAction(self):
        ''' Return optimized action from current state '''

        return numpy.argmax(self.net.getQs(self.state))

    def epsGreedyAction(self, env):
        ''' Return eps-greedy action from current state '''

        self.eps -= (INITIAL_EPS - FINAL_EPS) / EPISODE
        return random.randint(0, env.action_space.n - 1) if random.random() < self.eps else self.optAction()
    
    def perceive(self, oldState, action, newState, reward, done):
        ''' Train '''

        self.replay.append({
            'state': oldState,
            'action': action,
            'Q': reward if done else reward + GAMMA * numpy.max(self.net.getQs(oldState))
        })
        if len(self.replay) > REPLAY_SIZE:
            self.replay.popleft()
        batch = random.sample(self.replay, min(len(self.replay), BATCH_SIZE))
        self.net.feed([item['state'] for item in batch], [item['action'] for item in batch], [item['Q'] for item in batch])

def mainLoop(env, agent):
    ''' Main loop
        Left `env` and `agent` as parameters to make debugging easier '''

    for i in range(EPISODE):
        agent.reset()
        for j in range(STEP):
            oldState = agent.state
            action = agent.epsGreedyAction(env)
            newState, reward, done, info = env.step(action)
            agent.perceive(oldState, action, newState, reward, done)
            if done:
                break
            agent.state = newState

        # Tests
        if i % TEST_INTERVAL == 0:
            totReward = 0
            for case in range(TEST_CASES):
                agent.reset()
                success = True
                for j in range(STEP):
                    env.render()
                    oldState = agent.state
                    action = agent.optAction()
                    newState, reward, done, info = env.step(action)
                    totReward += reward
                    if done:
                        success = False
                        print("  Failed in %s steps"%(j))
                        break
                    agent.state = newState
                if success:
                    print("  Success")
            avgReward = totReward / TEST_CASES
            print("In %d-th episode, avgReward = %f"%(i, avgReward))

if __name__ == '__main__':

    env = gym.make(ENV_NAME)
    agent = Agent(env)
    mainLoop(env, agent)


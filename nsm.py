# NSM sequence example


import numpy as np
import matplotlib.pyplot as plt
import difflib
from random import randrange


class Experience:  # Stores one experience element of the  agent
    def __init__(self, action, reward, observation):
        self.action = action
        self.reward = reward
        self.observation = observation
        self.q = 0

    def __eq__(self, s):
        return (self.action == s.action) and (self.reward == s.reward) and (self.observation == s.observation)


class Enviroment:  # Maze.
    def __init__(self):
        self.Map = [[7, 8, 9, 10, 11], [4, -1, 5, -1, 6], [1, -1, 2, -1, 3]]
        # self.Obs = [[9, 5, 1, 5, 3], [10, -1, 10, -1, 10], [14, -1, 14, -1, 14]]
        self.Obs = [[1, 2, 3, 4, 5], [6, -1, 7, -1, 8], [9, -1, 10, -1, 11]]
        self.states = None
        self.actions = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        self.location_s = 2
        self.location_h = 0
        self.reward = 0
        self.pre = None
        self.iter = 0
    def printMap(self):
        for i in range(len(self.Map)):
            line = ""

            for j in range(len(self.Map[0])):
                if self.Map[i][j] == -1:
                    line += 'x'
                else:
                    if i == self.location_s and j == self.location_h:
                        line += 'o'
                    else:
                        line += '-'

            print(line)

        print('*****************')

    def reset(self):
        self.location_h = 0
        self.location_s = 2

    def is_valid(self, action):
        chosen_action = self.actions[action]
        return self.checkbounder(action) and self.Map[self.location_s + chosen_action[0]][
            self.location_h + chosen_action[1]] is not -1  # check if valid step

    def step(self, action):
        self.iter += 1
        # print('now : ', self.location_s, self.location_h)
        chosen_action = self.actions[action]
        observation = None


        if self.is_valid(action):
            self.location_s += chosen_action[0]
            self.location_h += chosen_action[1]

        if self.location_s == 2 and self.location_h == 4:  # set the reward
            self.reward = 100
            # print("yes")
        else:
            self.reward = 0

        observation = self.Obs[self.location_s][self.location_h]  # get the observation
        if self.iter % 2 == 0:
            if observation == self.pre:
                self.reward = -1
            self.iter = 0
            self.pre = observation

        self.printMap()

        return observation, self.reward

    # else:
    # return self.step(randrange(4))

    def checkbounder(self, action):
        if self.location_s + self.actions[action][0] < np.shape(self.Map)[0] and self.location_h + self.actions[action][
            1] < np.shape(self.Map)[1] and self.location_s + self.actions[action][0] >= 0 and self.location_h + \
                self.actions[action][1] >= 0:
            return True
        else:
            return False

    def sample(self):

        return randrange(4)


class Chain:  # Full sequence of experiences by the agent
    def __init__(self, N):
        self.N = N
        self.container = []

    # self.container = [Experience(1,0,14),Experience(1,0,10)]

    def add(self, experience):
        if len(self.container) < self.N:
            self.container.append(experience)
        else:
            self.container.pop(0)
            self.container.append(experience)

    def getKNeighbours(self, k):
        m = len(self.container)

        if m < 2:
            return []

        n = np.zeros((m, m))
        for i in range(len(n[0])):
            for j in range(len(n[1])):
                n[i][j] = randrange(3)

        for i in range(m):
            for j in range(m):
                if i > 0 and j > 0:
                    si = self.container[i - 1]
                    sj = self.container[j - 1]

                    if si == sj:
                        n[i, j] = 1 + n[i - 1, j - 1]
                    else:
                        n[i, j] = 0
                else:
                    n[i, j] = 1

        return [self.container[i] for i in range(len(n[-1]) - 1) if n[-1][i] >= k]


class Task:
    def __init__(self, k=3, N=500, beta=0.2, gamma=0.99, e=1.):
        self.k = k
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.e = e
        self.chain = Chain(self.N)
        # self.chain = self.readEXP()
        self.env = Enviroment()
        self.performance = []
        self.decay = 25.0
        self.min_e = 0.01
        self.min_beta = 0.15
        self.f = lambda x, min_x: max(min_x, min(1.0, 1.0 - np.log10((x + 1) / self.decay)))
        self.total = 0
    def saveEXP(self):
        a = []
        r = []
        o = []
        for exp in self.chain.container:
            a.append(exp.action)
            r.append(exp.reward)
            o.append(exp.observation)
        with open("action_list.txt", "w") as f:
            for temp in a:
                f.write(str(temp) + "\n")
        with open("reward_list.txt", "w") as f:
            for temp in r:
                f.write(str(temp) + "\n")
        with open("observation_list.txt", "w") as f:
            for temp in o:
                f.write(str(temp) + "\n")

    def readEXP(self):
        a = []
        r = []
        o = []
        with open("action_list.txt", "r") as f:
            for line in f:
                a.append(int(line.strip()))
        with open("reward_list.txt", "r") as f:
            for line in f:
                r.append(int(line.strip()))
        with open("observation_list.txt", "r") as f:
            for line in f:
                o.append(int(line.strip()))

        mChain = Chain(self.N)
        for i in range(len(a)):
            exp = Experience(a[i], r[i], o[i])
            mChain.add(exp)

        return mChain

    def getQTable(self, similar_states):
        Q = np.zeros((4, 1))
        counts = np.zeros((4, 1))

        for s in similar_states:
            Q[s.action] = Q[s.action] + s.q
            counts[s.action] += 1

        for i in range(4):
            if counts[i] > 0:
                Q[i] /= float(counts[i])
            else:
                Q[i] = 0

        return Q

    def getVotingStates(self, action, similar_states):
        voting_states = []

        for s in similar_states:
            if s.action == action:
                voting_states.append(s)

        return voting_states

    def nextAction(self, Q):

        # Make the decision using the e-greedy policy
        if np.random.random() < self.e:
            action = self.env.sample()
        else:
            action = np.argmax(Q)

        Qmax = np.max(Q)

        return action, Qmax

    def run(self, numEpisodes):
        # self.chain.add(Experience(3,0,3))
        # self.chain.add(Experience(2,0,6))
        # self.chain.add(Experience(2,0,11))
        # self.chain.add(Experience(1,0,10))
        # self.chain.add(Experience(1,0,9))
        # self.chain.add(Experience(3,0,5))
        # self.chain.add(Experience(3,1,2))
        for j in range(numEpisodes):
            self.env.reset();
            steps = 0
            done = False

            while not done:
                # We get the similar experiences
                # They are instances of Experience
                similar_states = self.chain.getKNeighbours(self.k)
                # print('Similar states: ')

                # for i in similar_states:
                #     print(i.action, i.reward, i.observation)

                Q = self.getQTable(similar_states)

                action, Qmax = self.nextAction(Q)

                while not self.env.is_valid(action):
                    action, Qmax = self.nextAction(Q)

                # get observation from the world
                observation, reward = self.env.step(action)

                if reward > 0:
                    # print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
                    done = True

                # print(self.env.location_s,self.env.location_h,reward)
                ee = Experience(action, reward, observation)
                # print('Current experience: {}, {}, {}'.format(ee.action, ee.reward, ee.observation))
                self.chain.add(ee)

                voting_states = self.getVotingStates(action, similar_states)

                # Update the Q values
                for s in voting_states:
                    s.q = (1 - self.beta) * s.q + self.beta * (reward + self.gamma * Qmax)

                steps += 1

            self.e = self.f(j, self.min_e)
            self.beta = self.f( j, self.min_beta )

            self.performance.append(steps)
            self.total += steps
            
        self.saveEXP()
        print('k = ',self.k,' total = ',self.total)
        plt.plot(self.performance,label = 'k = ' + str(self.k))
        plt.title('Performance (steps till reward)')
        plt.show()


if __name__ == "__main__":
    for i in range(6,7):
        task = Task(k = i)
        task.run(300)
    # plt.legend()
    # plt.show()





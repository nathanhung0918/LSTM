import gym
import numpy as np
import matplotlib.pyplot as plt


class MChain:
    def __init__(self, size, chain=None):
        self.size = size
        self.chain = chain

    def create(self, Obj):
        if self.chain == None:
            return [Obj() for i in range(self.size)]
        else:
            return [self.chain.create(Obj) for i in range(self.size)]

        # LocalDemon: one for each cell. i.e. states


class LocalDemon:
    def __init__(self):
        self.Q = np.array([0.0, 0.0])
        self.gamma = 1.0
        self.action = None

    def getMax(self):
        return np.max(self.Q), np.argmax(self.Q)

    def act(self, env, e):
        if np.random.random_sample() < e:
            self.action = env.action_space.sample()
        else:
            _, self.action = self.getMax()

    def update(self, alpha, reward, demon):
        Qa, _ = demon.getMax()
        # if reward == 1:
        # print("reward",reward)
        # print("action",self.action)

        self.Q[self.action] += alpha * (reward + self.gamma * Qa - self.Q[self.action])
        # print("Q after update*****",self.Q)
        # else:
        # The wrong action's value is zeroed
        # self.Q[self.action] = 0.0


class GlobalDemon:

    def __init__(self, env):
        self.currentDemon = None
        self.env = env
        lc = MChain(3, MChain(6, MChain(3, MChain(14, None))))
        self.localDemons = lc.create(LocalDemon)

    def getLocalDemon(self, state):
        xThresholds = np.array([-4.8])
        # thetaThresholds = np.array([-24.0, -16.0, -8.0, 0.0, 8.0, 16.0])
        thetaThresholds = np.arange(-24,17,8)
        dxThresholds = np.array([-np.inf])
        # dthetaThresholds = np.array([-np.inf, -50.0,-41.7,-33.3,-25.01,-16.68,-8.35,-0.02,8.31,16.64,25.0,33.3,41.63,50.0])
        dthetaThresholds = np.concatenate(((np.array([-np.inf]),np.arange(-50,51,8))),axis=0)
        # dthetaThresholds = np.arange(-50,51,20)
        x, dx, theta, dtheta = state
        theta = theta * 180.0 / np.pi
        dtheta = dtheta * 180.0 / np.pi

        i = np.where(xThresholds < x)[0][-1] #cart position
        j = np.where(thetaThresholds < theta)[0][-1] #cart velocity
        k = np.where(dxThresholds < dx)[0][-1]  #pole angle
        l = np.where(dthetaThresholds < dtheta)[0][-1] # pole velocity at tip
        # print(i,j,k,l)
        return self.localDemons[i][j][k][l]

    def act(self, state, e):
        self.currentDemon = self.getLocalDemon(state)
        # print("Q before update", self.currentDemon.Q)
        self.currentDemon.act(self.env, e)
        return self.currentDemon.action

    def update(self, alpha, state, reward):
        newDemon = self.getLocalDemon(state)
        self.currentDemon.update(alpha, reward, newDemon)


class Task:

    def __init__(self, nEpisodes, T):
        self.decay = 25.0
        self.min_e = 0.01
        self.min_alpha = 0.15
        self.nEpisodes = nEpisodes
        self.T = T

    def plot(self):
        plt.plot(self.rewards)
        plt.title('Performance (survive time)')
        plt.xlabel('iteration')
        plt.ylabel('taken steps')
        plt.show()

    def run(self):

        self.rewards = []
        e = 1.
        alpha = 1.
        env = gym.make('CartPole-v1')
        demon = GlobalDemon(env)

        f = lambda x, min_x: max(min_x, min(1.0, 1.0 - np.log10((x + 1) / self.decay)))

        for i in range(self.nEpisodes):
            observation = env.reset()
            total_reward = 0.0

            for t in range(self.T):
                # if i_episode > 690:
                # env.render()
                # print(observation)
                action = demon.act(observation, e)
                observation, reward, done, info = env.step(action)
                total_reward += reward

                if done:
                    # Punishing wrong action
                    # demon.update( alpha, observation, 0 )
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
                else:
                    demon.update(alpha, observation, reward)

            # e = f(i, self.min_e)
            # alpha = f(i, self.min_alpha)
            self.rewards.append(total_reward)

        env.close()


if __name__ == "__main__":
    task = Task(1000, 501)
    task.run()
    task.plot()
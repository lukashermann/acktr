import gym
from utils import *
import utils
from filters import ZFilter, IdentityFilter, ClipFilter
from normalized_env import NormalizedEnv # used only for rescaling actions
from rgb_env import RGBEnv
from jaco_pixel_env import JacoPixelEnv
from jaco_depth_env import JacoDepthEnv
from jaco_combi_env import JacoCombiEnv
import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
import tempfile
import sys
import argparse
import kfac
import shutil
import pickle
import datetime

parser = argparse.ArgumentParser(description="Run commands")
# GENERAL HYPERPARAMETERS
parser.add_argument('-e', '--env-id', type=str, default="Pendulum-v0",
                    help="Environment id")
parser.add_argument('-mt', '--max-timesteps', default=100000000, type=int,
                    help="Maximum number of timesteps")
parser.add_argument('-tpb', '--timesteps-per-batch', default=1000, type=int,
                    help="Minibatch size")
parser.add_argument('-g', '--gamma', default=0.99, type=float,
                    help="Discount Factor")
parser.add_argument('-l', '--lam', default=0.97, type=float,
                    help="Lambda value to reduce variance see GAE")
parser.add_argument('-s', '--seed', default=1, type=int,
                    help="Seed")
parser.add_argument('--log-dir', default="./logs/", type=str,
                    help="Folder to save")
# NEURAL NETWORK ARCHITECTURE
parser.add_argument('--weight-decay-fc', default=3e-4, type=float, help="weight decay for fc layer")
parser.add_argument('--weight-decay-conv', default=4e-3, type=float, help="weight decay for conv layer")
parser.add_argument('--use-pixels', default=False, type=bool, help="use rgb instead of low dim state rep")
# GENERAL KFAC arguments
parser.add_argument('--async-kfac', default=True, type=bool, help="use async version")
# POLICY HYPERPARAMETERS
parser.add_argument('--use-adam', default=False, type=bool, help="use adam for actor")
parser.add_argument('--use-sgd', default=False, type=bool, help="use sgd with momentum for actor")
parser.add_argument('--adapt-lr', default=True, type=bool, help="adapt lr")
parser.add_argument('--upper-bound-kl', default=False, type=bool, help="upper bound kl")
parser.add_argument('--lr', default=0.03, type=float, help="Learning Rate")
parser.add_argument('--mom', default=0.9, type=float, help="Momentum")
parser.add_argument('--kl-desired', default=0.001, type=float, help="desired kl div")
parser.add_argument('--kfac-update', default=2, type=int,
                    help="Update Fisher Matrix every number of steps")
parser.add_argument('--cold-iter', default=1, type=int,
                    help="Number of cold iterations using sgd")
parser.add_argument('--epsilon', default=1e-2, type=float, help="Damping factor")
parser.add_argument('--stats-decay', default=0.99,type=float, help="decay running average of stats factor")
# VALUE FUNCTION HYPERPARAMETERS
parser.add_argument('--use-adam-vf', default=False, type=bool, help="use adam for vf")
parser.add_argument('--use-sgd-vf', default=False, type=bool, help="use sgd with momentum for vf")
parser.add_argument('--lr-vf', default=0.003, type=float, help="Learning Rate vf")
parser.add_argument('--cold-lr-vf', default=0.001, type=float, help="Learning Rate vf")
parser.add_argument('--mom-vf', default=0.9, type=float, help="Momentum")
parser.add_argument('--kl-desired-vf', default=0.3, type=float, help="desired kl div")
parser.add_argument('--epsilon-vf', default=0.1, type=float, help="Damping factor")
parser.add_argument('--stats-decay-vf', default=0.95, type=float, help="Damping factor")
parser.add_argument('--kfac-update-vf', default=2, type=int,
                    help="Update Fisher Matrix every number of steps")
parser.add_argument('--cold-iter-vf', default=50, type=int,
                    help="Number of cold iterations using sgd")
parser.add_argument('--train-iter-vf', default=25, type=int,
                    help="Number of cold iterations using sgd")
parser.add_argument('--moving-average-vf', default=0.0, type=float,
                    help="Moving average of VF parameters")
parser.add_argument('--load-model', default=True, type=bool,
                    help="Load trained model")
parser.add_argument('--load-dir', default="/home/hermannl/master_project/git/emansim/acktr/logs/JacoPixel-v1_combi3/openai-2017-11-07-13-41-11", type=str,
                    help="Folder to load from")
parser.add_argument('--is-rgb', default=True, type=bool,
                    help="Use RGB")
parser.add_argument('--is-depth', default=False, type=bool,
                    help="Use Depth Image")

class AsyncNGAgent(object):

    def __init__(self, env, args):
        self.env = env
        self.config = config = args
        self.config.max_pathlength = 150 #env._spec.tags.get('wrapper_config.TimeLimit.max_episode_steps') or 1000
        # set weight decay for fc and conv layers
        utils.weight_decay_fc = self.config.weight_decay_fc
        utils.weight_decay_conv = self.config.weight_decay_conv

        # hardcoded for now
        if self.config.use_adam:
            self.config.kl_desired = 0.002
            self.lr = 1e-4
        env_description_str = self.config.env_id
        env_description_str += "_test1"
        self.config.log_dir = os.path.join("test_logs/",env_description_str,
        datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S") )

        # print all the flags
        print '##################'
        # save hyperparams to txt file
        hyperparams_txt = ""
        for key,value in vars(self.config).iteritems():
            print key, value
            hyperparams_txt = hyperparams_txt + "{} {}\n".format(key, value)
        if os.path.exists(self.config.log_dir):
            shutil.rmtree(self.config.log_dir)
        os.makedirs(self.config.log_dir)
        txt_file = open(os.path.join(self.config.log_dir, "hyperparams.txt"), "w")
        txt_file.write(hyperparams_txt)
        txt_file.close()
        print (self.config.log_dir)
        print '##################'
        print("Observation Space Pixel", env.observation_space_pix)
        print("Observation Space State Space", env.observation_space_ss)
        print("Action Space", env.action_space)
        config_tf = tf.ConfigProto(intra_op_parallelism_threads=1)
        config_tf.gpu_options.allow_growth=True # don't take full gpu memory
        self.session = tf.Session(config=config_tf)
        """from tensorflow.python import debug as tf_debug
        self.session = tf_debug.LocalCLIDebugWrapperSession(self.session)
        self.session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)"""
        self.train = False
        self.solved = False
        self.obs_pix_shape = obs_pix_shape = list(env.observation_space_pix.shape)
        self.obs_ss_shape = obs_ss_shape = list(env.observation_space_ss.shape)
        self.prev_obs_pix = np.zeros([1] + list(obs_pix_shape))
        self.prev_obs_ss = np.zeros([1] + list(obs_ss_shape))
        self.prev_action = np.zeros((1, env.action_space.shape[0]))
        obs_pix_shape[-1] *= 2 # include previous frame in a state
        obs_ss_shape[-1] *= 2 # include previous frame in a state
        self.obs_pix = obs_pix = tf.placeholder(
                dtype, shape=[None] + obs_pix_shape, name="obs_pix")
        self.obs_ss = obs_ss = tf.placeholder(
                dtype, shape=[None, 2*env.observation_space_ss.shape[0] + env.action_space.shape[0]], name="obs_ss")

        self.action = action = tf.placeholder(dtype, shape=[None, env.action_space.shape[0]], name="action")
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
        self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[None, env.action_space.shape[0]*2], name="oldaction_dist")


        self.ob_pix_filter = IdentityFilter()
        self.reward_filter = ZFilter((1,), demean=False, clip=10)
        self.ob_ss_filter = ZFilter((env.observation_space_ss.shape[0],), clip=5)

        # Create summary writer
        self.summary_writer = tf.summary.FileWriter(self.config.log_dir)

        self.animate = True
        if self.animate:
            self.img_save_path = self.config.log_dir + "/imgs/"
            os.mkdir(self.img_save_path)
        self.save_frames = False


    def act(self, obs_pix, obs_ss, *args):
        obs_ss = self.ob_ss_filter(obs_ss, update=self.train)
        obs_pix = self.ob_pix_filter(obs_pix)
        obs_ss = np.expand_dims(obs_ss, 0)
        obs_pix = np.expand_dims(obs_pix, 0)
        obs_pix_new = np.concatenate([obs_pix, self.prev_obs_pix], -1)
        obs_ss_new = np.concatenate([obs_ss, self.prev_obs_ss, self.prev_action], 1)

        action_dist_n = self.session.run(self.action_dist_n, {self.obs_pix: obs_pix_new,self.obs_ss: obs_ss_new})

        """
        if self.train:
            action = np.float32(gaussian_sample(action_dist_n, self.env.action_space.shape[0]))
        else:
            action = np.float32(deterministic_sample(action_dist_n, self.env.action_space.shape[0]))
        """
        action = np.float32(deterministic_sample(action_dist_n, self.env.action_space.shape[0]))

        self.prev_action = np.expand_dims(np.copy(action),0)
        self.prev_obs_pix = obs_pix
        self.prev_obs_ss = obs_ss
        return action, action_dist_n, np.squeeze(obs_pix_new), np.squeeze(obs_ss_new)

    def deploy(self):
        config = self.config
        numeptotal = 0
        i = 0
        total_timesteps = 0
        benchmark_results = []
        benchmark_results.append({"env_id": config.env_id})

        # Create saver

        #self.train = False
        self.saver = tf.train.import_meta_graph('{}/model.ckpt.meta'.format(config.load_dir))
        self.saver.restore(self.session, \
            tf.train.latest_checkpoint("{}".format(config.load_dir)))

        ob_filter_path = os.path.join(config.load_dir, "ob_filter.pkl")
        with open(ob_filter_path, 'rb') as ob_filter_input:
            self.ob_ss_filter = pickle.load(ob_filter_input)

        print ("Loaded Model")

        policy_vars = []
        # recreate policy net
        for j,var in enumerate(tf.global_variables()):
            if var.name.startswith("policy"):
                policy_vars.append(var)
                print (var.name)
                print (self.session.run(var, feed_dict={}).shape)
                """mat = var.eval(session=self.session)
                with open(os.path.join(self.config.log_dir, "w" + str(j) + ".npz"), "w") as outfile:
                    np.save(outfile, mat)"""

        self.action_dist_n = load_policy_net_combi42(self.obs_pix, self.obs_ss, policy_vars, [64,64], [True, True], env.action_space.shape[0])

        while total_timesteps < self.config.max_timesteps:
            # save frames
            self.save_frames = False
            self.iteration = i
            if (i % 1 == 0) and self.animate:
                print("true")
                os.mkdir(self.img_save_path + "iter_" + str(i) )
                self.save_frames = True

            # Generating paths.
            print("Rollout")
            t1_rollout = time.time()
            paths, timesteps_sofar = rollout(
                self.env,
                self,
                config.max_pathlength,
                config.timesteps_per_batch)
            total_timesteps += timesteps_sofar
            t2_rollout = time.time()
            print ("Time for rollout")
            print (t2_rollout - t1_rollout)
            start_time = time.time()

            # write results to monitor.json
            for path in paths:
                curr_result = {}
                curr_result["l"] = len(path["rewards"])
                curr_result["r"] = path["rewards"].sum()
                benchmark_results.append(curr_result)


            episoderewards = np.array(
                [path["rewards"].sum() for path in paths])

            print "\n********** Iteration %i ************" % i
            if episoderewards.mean() >= self.env._spec.reward_threshold:
                print "Solved Env"
                self.solved = True


            stats = {}
            numeptotal += len(episoderewards)
            stats["Total number of episodes"] = numeptotal
            stats["Average sum of rewards per episode"] = episoderewards.mean()
            for k, v in stats.iteritems():
                print(k + ": " + " " * (40 - len(k)) + str(v))

            i += 1


if __name__ == '__main__':

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    env = gym.make(args.env_id)
    if args.use_pixels:
        env = JacoCombiEnv(env, is_rgb=True, is_depth=True)
    else:
        env = NormalizedEnv(env)
    agent = AsyncNGAgent(env, args)
    agent.deploy()

import numpy as np
import tensorflow as tf
import random
import scipy.signal
import scipy.optimize
import sys
import kfac
import json
import time
from PIL import Image
import os
import shutil
import copy
import cv2

dtype = tf.float32
weight_decay_fc = 0.0
weight_decay_conv = 0.0

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def save_ob(ob, folder, timesteps_sofar):
    Image.fromarray((copy.deepcopy(ob) * 255.).astype(np.uint8)).save(folder + '/ob_{}.jpg'.format(timesteps_sofar))

def save_obs(ob_raw, ob, folder, timesteps_sofar):
    #print(ob.shape)

    Image.fromarray((copy.deepcopy(ob_raw)).astype(np.uint8)).save(folder + '/ob_raw_{}.jpg'.format(timesteps_sofar))
    #Image.fromarray((copy.deepcopy(ob) * 255.).astype(np.uint8)).save(folder + '/ob_{}.jpg'.format(timesteps_sofar))
    cv2.imwrite(folder + '/ob_{}.jpg'.format(timesteps_sofar),ob*255)


def remkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)

def save_intermediate_observations(obs, folder, timesteps_sofar):
    path = os.path.join(folder, 'intermediate_obs_{}'.format(timesteps_sofar))
    os.mkdir(path)
    for j,int_ob in enumerate(obs):
        for i in range(int_ob.shape[3]):
            img = int_ob[0,:,:,i]*255
            img = cv2.resize(img, (200,200), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(path + '/intermediate_ob_{}_{}_{}.jpg'.format(timesteps_sofar,j,i),img)
# Sample only 1 episode
def load_rollout(env, agent, max_pathlength, n_timesteps, save=False, save_dir="./dummy/"):
    paths = []
    timesteps_sofar = 0

    obs, actions, rewards, rewards_filtered, action_dists = [], [], [], [], []
    ob_raw, ob = env.reset()
    if save and agent.config.use_pixels:
        folder = os.path.join(save_dir, "episode_{}".format(agent.iter))
        # create folder if doesn't exists (remove if exists)
        if agent.iter == 0:
            remkdir(save_dir)
        remkdir(folder)
        save_obs(ob_raw, ob, folder, timesteps_sofar)


    agent.prev_action *= 0.0
    agent.prev_obs *= 0.0
    terminated = False

    for _ in xrange(max_pathlength):
        action, action_dist, ob, intermediate_obs = agent.act(ob)
        obs.append(ob)
        actions.append(action)
        action_dists.append(action_dist)
        res = env.step(action)
        timesteps_sofar += 1
        reward_filtered = agent.reward_filter(np.asarray([res[2]]))[0]
        ob_raw = res[0]
        ob = res[1]
        rewards.append(res[2])
        rewards_filtered.append(reward_filtered)
        if save and agent.config.use_pixels:
            folder = os.path.join(save_dir, "episode_{}".format(agent.iter))
            save_obs(ob_raw, ob, folder, timesteps_sofar)
            # save intermediate_obs
            save_intermediate_observations(intermediate_obs, folder, timesteps_sofar)
        if res[3]:
            terminated = True
            break

    path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
            "action_dists": np.concatenate(action_dists),
            "rewards": np.array(rewards),
            "rewards_filtered": np.array(rewards_filtered),
            "actions": np.array(actions),
            "terminated": terminated,}
    paths.append(path)
    agent.prev_action *= 0.0
    agent.prev_obs *= 0.0
    timesteps_sofar += len(path["rewards"])
    return paths, timesteps_sofar

def rollout(env, agent, max_pathlength, n_timesteps):
    paths = []
    timesteps_sofar = 0
    while timesteps_sofar < n_timesteps:
        if agent.config.use_pixels:
            obs_pix, obs_ss, actions, rewards, rewards_filtered, action_dists = [], [], [], [], [], []
            ob_pix, ob_ss = env.reset()
        else:
            obs_ss, actions, rewards, rewards_filtered, action_dists = [], [], [], [], []
            ob_ss = env.reset()
        agent.prev_action *= 0.0
        agent.prev_obs_ss *= 0.0
        terminated = False

        for j in xrange(max_pathlength):
            if agent.save_frames:
                frame = env.render(mode="rgb_array")

                cv2.imwrite(agent.img_save_path + "iter_"+str(agent.iteration)+"/img_" + str(j) + ".png", frame)
            if agent.config.use_pixels:
                action, action_dist, ob_pix, ob_ss = agent.act_combi(ob_pix, ob_ss)
                obs_pix.append(ob_pix)
            else:
                action, action_dist, ob_ss = agent.act_ss(ob_ss)
            obs_ss.append(ob_ss)
            actions.append(action)
            action_dists.append(action_dist)
            res = env.step(action)
            reward_filtered = agent.reward_filter(np.asarray([res[1]]))[0]
            if agent.config.use_pixels:
                ob_pix = res[0]
                ob_ss = res[4]
            else:
                ob_ss = res[0]
            rewards.append(res[1])
            rewards_filtered.append(reward_filtered)
            if res[2]:
                terminated = True
                break
        agent.save_frames = False
        if agent.config.use_pixels:
            path = {"obs_pix": np.concatenate(np.expand_dims(obs_pix, 0)),
                    "obs_ss": np.concatenate(np.expand_dims(obs_ss, 0)),
                    "action_dists": np.concatenate(action_dists),
                    "rewards": np.array(rewards),
                    "rewards_filtered": np.array(rewards_filtered),
                    "actions": np.array(actions),
                    "terminated": terminated,}
        else:
            path = {"obs_ss": np.concatenate(np.expand_dims(obs_ss, 0)),
                    "action_dists": np.concatenate(action_dists),
                    "rewards": np.array(rewards),
                    "rewards_filtered": np.array(rewards_filtered),
                    "actions": np.array(actions),
                    "terminated": terminated,}
        paths.append(path)
        agent.prev_action *= 0.0
        if agent.config.use_pixels:
            agent.prev_obs_pix *= 0.0
        agent.prev_obs_ss *= 0.0
        timesteps_sofar += len(path["rewards"])
    return paths, timesteps_sofar

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def linear(x, size, name, initializer=None, bias_init=0, weight_loss_dict=None, reuse=None):
#    assert len(name.split('/')) == 2 # make sure that name has format policy/l1 or vf/l1

    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable("w", [x.get_shape()[1], size], initializer=initializer)
        b = tf.get_variable("b", [size], initializer=tf.constant_initializer(bias_init))

        if weight_decay_fc > 0.0 and weight_loss_dict is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(w), weight_decay_fc, name='weight_decay_loss')
            if weight_loss_dict is not None:
                weight_loss_dict[w] = weight_decay_fc
                weight_loss_dict[b] = 0.0
            tf.add_to_collection(name.split('/')[0] + '_' + 'losses', weight_decay)

        return tf.nn.bias_add(tf.matmul(x, w), b)

def linearnobias(x, size, name, initializer=None, weight_loss_dict=None, reuse=None):
    #assert len(name.split('/')) == 2 # make sure that name has format policy/l1 or vf/l1
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)

        if weight_decay_fc > 0.0 and weight_loss_dict is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(w), weight_decay_fc, name='weight_decay_loss')
            if weight_loss_dict is not None:
                weight_loss_dict[w] = weight_decay_fc
            tf.add_to_collection(name.split('/')[0] + '_' + 'losses', weight_decay)

        return tf.matmul(x, w)

def gaussian_sample(action_dist, action_size):
    return np.random.randn(action_size) * action_dist[0,action_size:] + action_dist[0,:action_size]

def deterministic_sample(action_dist, action_size):
    return action_dist[0,:action_size]

# returns mean and std of gaussian distribution
def get_moments(action_dist, action_size):
    mean = tf.reshape(action_dist[:, :action_size], [tf.shape(action_dist)[0], action_size])
    std = (tf.reshape(action_dist[:, action_size:], [tf.shape(action_dist)[0], action_size]))
    return mean, std


def loglik(action, action_dist, action_size):
    mean, std = get_moments(action_dist, action_size)
    return -0.5 * tf.reduce_sum(tf.square((action-mean) / std),reduction_indices=-1) \
            -0.5 * tf.log(2.0*np.pi)*action_size - tf.reduce_sum(tf.log(std),reduction_indices=-1)

def kl_div(action_dist1, action_dist2, action_size):
    mean1, std1 = get_moments(action_dist1, action_size)
    mean2, std2 = get_moments(action_dist2, action_size)
    numerator = tf.square(mean1 - mean2) + tf.square(std1) - tf.square(std2)
    denominator = 2 * tf.square(std2) + 1e-8
    return tf.reduce_sum(
        numerator/denominator + tf.log(std2) - tf.log(std1),reduction_indices=-1)

def entropy(action_dist, action_size):
    _, std = get_moments(action_dist, action_size)
    return tf.reduce_sum(tf.log(std),reduction_indices=-1) + .5 * np.log(2*np.pi*np.e) * action_size

def conv2d_loaded(x, weights, biases, num_filters, filter_size=(3, 3), stride=(1, 1), pad="SAME"):
    filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
    stride_shape = [1, stride[0], stride[1], 1]

    return tf.nn.bias_add(tf.nn.conv2d(x, weights, stride_shape, pad), biases)

# Bits and pieces taken from Jimmy and universe-starter-agent
def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", initializer=None, bias_init=0, weight_loss_dict=None, reuse=None):
    with tf.variable_scope(name, reuse=reuse):
        if initializer == None:
            stddev = 0.01
            initializer = tf.random_normal_initializer(stddev=stddev)

        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
        stride_shape = [1, stride[0], stride[1], 1]

        weights = tf.get_variable('weights', filter_shape,
                                  initializer=initializer)
        biases = tf.get_variable(
            'biases', [num_filters], initializer=tf.constant_initializer(0.))

        if weight_decay_conv > 0.0 and weight_loss_dict is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(weights), weight_decay_conv, name='weight_decay_loss')
            if weight_loss_dict is not None:
                weight_loss_dict[weights] = weight_decay_conv
                weight_loss_dict[biases] = 0.0
            tf.add_to_collection(name.split('/')[0] + '_' + 'losses', weight_decay)

        return tf.nn.bias_add(tf.nn.conv2d(x, weights, stride_shape, pad), biases)

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(grad, [numel(v)])
                      for (v, grad) in zip(var_list, grads)], 0)


class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def explained_variance(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

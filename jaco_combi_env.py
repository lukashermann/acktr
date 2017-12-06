import numpy as np
from scipy.misc import imresize
from gym.spaces import Discrete, Box, Tuple
from gym import Env
import cv2

class JacoCombiEnv(Env):
    def __init__(self, env, is_rgb=False, is_depth=False):
        self._env = env
        self.is_rgb = is_rgb
        self.is_depth = is_depth
        self.width = 42
        self.num_channels = 1
        if self.is_rgb:
            self.num_channels = 3
        if self.is_depth:
            self.num_channels += 1
        self._observation_space_pix = Box(low=0.0, high=1.0, shape=(self.width, self.width, self.num_channels))
        self._observation_space_ss = self._env.unwrapped.observation_space
        self._spec = self._env.unwrapped.spec
        self._spec.reward_threshold = self._spec.reward_threshold or float('inf')

    @property
    def action_space(self):
        if isinstance(self._env.action_space, Box):
            ub = np.ones(self._env.action_space.shape)
            return Box(-1 * ub, ub)
        return self._env.action_space

    @property
    def observation_space_pix(self):
        return self._observation_space_pix

    @property
    def observation_space_ss(self):
        return self._observation_space_ss

    # Taken from universe-starter-agent
    def _process_frame84(self, frame):
        #frame = frame[20:,10:190]
        # Resize by half, then down to 42x42 (essentially mipmapping). If
        # we resize directly we lose pixels that, when mapped to 42x42,
        # aren't close enough to the pixel boundary.
        frame = cv2.resize(frame, (120, 120)) # 80, 80
        frame = cv2.resize(frame, (84, 84)) # 42, 42
        if self.is_rgb is False:
            frame = frame.mean(2)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        if self.is_rgb is False:
            frame = np.reshape(frame, [84, 84, 1])
        else:
            frame = np.reshape(frame, [84, 84, 3])
        return frame

    def preprocessRgb(self,img,height, width):
        # resize from 200x200 to heightxwidth
        # convert to grayscale
        img = cv2.resize(img, (height*2, width*2))
        img = cv2.resize(img, (height,width))
        if not self.is_rgb:
            img = img[:,:,::-1]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
        img = img.astype(np.float32)
        img*= (1. / 255.)
        if self.is_rgb is False:
            img = np.reshape(img, [height, width, 1])
        #cv2.imshow("rgb", img)
        #cv2.waitKey(1)
        return img
    def preprocessDepth(self,img, height, width):
        #img /= img.max()
        img = np.clip(img, 0,0.003)
        img /= 0.003 # not always accurate, use .max() to be safe

        img = cv2.resize(img, (height, width), interpolation = cv2.INTER_NEAREST)
        #cv2.imshow("depth", img)
        #cv2.waitKey(1)
        return img

    def preprocessMujocoRgbd(self,ob, height, width):
        rgb = self.preprocessRgb(ob[0], height, width)
        depth = self.preprocessDepth(ob[1], height, width)
        frame = np.zeros((height, width,self.num_channels))
        if self.is_depth:
            frame[:,:,:-1] = rgb
            frame[:,:,-1] = depth
        else:
            frame = rgb
        return frame

    def reset(self, **kwargs):
        ob = self._env.reset(**kwargs)
        #frame = self._process_frame84(self._env.render('rgb_array'))
        frame = self.preprocessMujocoRgbd(ob, self.width, self.width)
        return frame, ob[2]

    def step(self, action):
        if isinstance(self._env.action_space, Box):
            # rescale the action
            lb = self._env.action_space.low
            ub = self._env.action_space.high
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action

        wrapped_step = self._env.step(scaled_action)
        ob, reward, done, info = wrapped_step
        next_frame = self.preprocessMujocoRgbd(ob, self.width, self.width)

        ob_ss = ob[2]

        return next_frame, reward, done, info, ob_ss

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def __getattr__(self, field):
        """
        proxy everything to underlying env
        """
        if hasattr(self._env, field):
            return getattr(self._env, field)
        raise AttributeError(field)

    def __repr__(self):
        if "object at" not in str(self._env):
            env_name = str(env._env)
        else:
            env_name = self._env.__class__.__name__
        return env_name

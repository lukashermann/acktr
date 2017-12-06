from utils import *
import utils

class VF_combi(object):
    coeffs = None

    def __init__(self, config, session):
        self.net = None
        self.config = config
        self.session = session
        # use exponential average when computing baseline
        self.averager = tf.train.ExponentialMovingAverage(decay=self.config.moving_average_vf)

    def init_vf(self,paths):
        featmat_pix = np.concatenate([self._features_rgb(path) for path in paths])

        featmat_ss = np.concatenate([self._features(path) for path in paths])
        return self.create_net(featmat_pix.shape[1:],[featmat_ss.shape[1]])

    def fc_net(self, x, weight_loss_dict=None, reuse=None):
        net = x
        hidden_sizes = [64,64]
        for i in range(len(hidden_sizes)):
            net = linear(net, hidden_sizes[i], "vf/l{}".format(i), initializer=normalized_columns_initializer(1.0), weight_loss_dict=weight_loss_dict, reuse=reuse)
            net = tf.nn.elu(net)

        net = linear(net, 1, "vf/value", initializer=None, weight_loss_dict=weight_loss_dict, reuse=reuse)
        net = tf.reshape(net, (-1, ))
        return net, weight_loss_dict

    def conv_net42(self, x, weight_loss_dict=None, reuse=None):

        # Conv Layers
        for i in range(2):
            x = tf.nn.elu(conv2d(x, 32, "vf/l{}".format(i), [3, 3], [2, 2], \
                initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))

        x = flatten(x)
        # One more linear layer
        x = linear(x, 256, "vf/l{}".format(i+1), \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.nn.elu(x)

        x = linear(x, 1, "vf/value", \
            initializer=ortho_init(1), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.reshape(x, (-1, ))

        return x, weight_loss_dict

    def conv_net_combi42(self, x_pix, x_ss, weight_loss_dict=None, reuse=None):
        # Conv Layers
        for i in range(2):
            x_pix = tf.nn.elu(conv2d(x_pix, 32, "vf/l{}".format(i), [3, 3], [2, 2], \
                initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))

        x_pix = flatten(x_pix)

        # Linear Layers
        hidden_sizes = [64,64]
        for i in range(len(hidden_sizes)):
            x_ss = linear(x_ss, hidden_sizes[i], "vf/l{}".format(i+2), initializer=normalized_columns_initializer(1.0), weight_loss_dict=weight_loss_dict, reuse=reuse)
            x_ss = tf.nn.elu(x_ss)

        x_pix = linear(x_pix, 256, "vf/l4", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x_pix = tf.nn.elu(x_pix)

        combined = tf.concat(1,[x_pix, x_ss])

        x = linear(combined, 128, "vf/l5", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.nn.elu(x)

        x = linear(x, 1, "vf/value", \
            initializer=ortho_init(1), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.reshape(x, (-1, ))

        return x, weight_loss_dict

    def conv_net_combi45(self, x_pix, x_ss, weight_loss_dict=None, reuse=None):
        # Conv Layers
        for i in range(3):
            x_pix = tf.nn.elu(conv2d(x_pix, 32, "vf/l{}".format(i), [3, 3], [2, 2],pad="VALID", \
                initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))
        x_pix = flatten(x_pix)

        # Linear Layers
        hidden_sizes = [64,64]
        for i in range(len(hidden_sizes)):
            x_ss = linear(x_ss, hidden_sizes[i], "vf/l{}".format(i+3), initializer=normalized_columns_initializer(1.0), weight_loss_dict=weight_loss_dict, reuse=reuse)
            x_ss = tf.nn.elu(x_ss)

        x = tf.concat(1,[x_pix, x_ss])

        x = linear(x, 256, "vf/l5", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.nn.elu(x)

        x = linear(x, 1, "vf/value", \
            initializer=ortho_init(1), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.reshape(x, (-1, ))

        return x, weight_loss_dict

    def conv_net84(self, x, weight_loss_dict=None, reuse=None):

        # Conv Layers
        x = tf.nn.elu(conv2d(x, 32, "vf/l0", [8, 8], [4, 4],pad="VALID", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))
        x = tf.nn.elu(conv2d(x, 32, "vf/l1", [4, 4], [2, 2],pad="VALID", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))
        x = tf.nn.elu(conv2d(x, 32, "vf/l2", [3, 3], [1, 1],pad="VALID", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))

        x = flatten(x)
        # One more linear layer
        x = linear(x, 512, "vf/l3", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.nn.elu(x)

        x = linear(x, 1, "vf/value", \
            initializer=ortho_init(1), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.reshape(x, (-1, ))

        return x, weight_loss_dict

    def conv_net63(self, x, weight_loss_dict=None, reuse=None):

        # Conv Layers
        x = tf.nn.elu(conv2d(x, 32, "vf/l0", [3, 3], [2, 2],pad="VALID", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))
        x = tf.nn.elu(conv2d(x, 32, "vf/l1", [3, 3], [2, 2],pad="VALID", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))
        x = tf.nn.elu(conv2d(x, 32, "vf/l2", [3, 3], [2, 2],pad="VALID", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))

        x = flatten(x)
        # One more linear layer
        x = linear(x, 256, "vf/l3", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.nn.elu(x)

        x = linear(x, 1, "vf/value", \
            initializer=ortho_init(1), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.reshape(x, (-1, ))

        return x, weight_loss_dict


    def create_net(self, shape_pix, shape_ss):
        self.x_pix = tf.placeholder(tf.float32, shape=[None] + list(shape_pix), name="x_pix")
        self.x_ss = tf.placeholder(tf.float32, shape=[None] + list(shape_ss), name="x_ss")

        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.vf_weight_loss_dict = {}
        with tf.name_scope('train_vf'):
            self.net, self.vf_weight_loss_dict = self.conv_net_combi42(self.x_pix, self.x_ss, self.vf_weight_loss_dict)

        self.bellman_error = (self.net - self.y)
        l2 = tf.reduce_mean(self.bellman_error * self.bellman_error)
        # get weight decay losses for value function
        vf_losses = tf.get_collection('vf_losses', None)

        self.loss = loss = l2 + tf.add_n(vf_losses)

        var_list_all = tf.trainable_variables()
        self.var_list = var_list = []
        for var in var_list_all:
            if "vf" in str(var.name):
                var_list.append(var)

        self.update_averages = self.averager.apply(self.var_list)

        # build test net with exponential moving averages for inference
        with tf.name_scope('test_vf'):
            self.test_net, _ = self.conv_net_combi42(self.x_pix, self.x_ss, None, reuse=True)

        if self.config.use_adam_vf:
            self.loss_fisher = None
        else:
            sample_net = self.net + tf.random_normal(tf.shape(self.net))
            self.loss_fisher = loss_fisher = tf.reduce_mean(tf.pow(self.net - tf.stop_gradient(sample_net), 2))

        return self.loss, self.loss_fisher, self.vf_weight_loss_dict

    def init_vf_train_op(self, loss_vf, loss_vf_sampled, wd_dict):
        if self.config.use_adam_vf:
            # 0.001
            self.update_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_vf)
            self.queue_runner = None
        elif self.config.use_sgd_vf:
            # 0.001*(1.-0.9), 0.9
            self.update_op = tf.train.MomentumOptimizer(0.001*(1.-0.9), 0.9).minimize(loss_vf)
            self.queue_runner = None
        else:
            self.update_op, self.queue_runner = kfac.KfacOptimizer(
                                             learning_rate=self.config.lr_vf,
                                             cold_lr=self.config.lr_vf/3.,
                                             momentum=self.config.mom_vf,
                                             clip_kl=self.config.kl_desired_vf,
                                             upper_bound_kl=False,
                                             epsilon=self.config.epsilon_vf,
                                             stats_decay=self.config.stats_decay_vf,
                                             async=self.config.async_kfac,
                                             kfac_update=self.config.kfac_update_vf,
                                             cold_iter=self.config.cold_iter_vf,
                                             weight_decay_dict=wd_dict).minimize(
                                                  loss_vf,
                                                  loss_vf_sampled,
                                                  self.var_list)

        with tf.control_dependencies([self.update_op]):
            self.train = tf.group(self.update_averages)

        return self.train, self.queue_runner

    def _features(self, path):
        o = path["obs_ss"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        act = path["action_dists"].astype('float32')
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 10.0
        ret = np.concatenate([o, act, al, np.ones((l, 1))], axis=1)
        return ret

    def _features_rgb(self, path):
        o = path["obs_pix"].astype('float32')
        return o

    def get_feed_dict(self, paths):
        featmat_pix = np.concatenate([self._features_rgb(path) for path in paths])
        featmat_ss = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        return {self.x_pix: featmat_pix,self.x_ss: featmat_ss, self.y: returns}

    def fit(self, paths):
        featmat_pix = np.concatenate([self._features_rgb(path) for path in paths])
        featmat_ss = np.concatenate([self._features(path) for path in paths])
        if self.net is None:
            self.create_net(featmat_pix.shape[1:],featmat_ss.shape[1:])
        returns = np.concatenate([path["returns"] for path in paths])

        self.session.run(self.train, {self.x_pix: featmat_pix,self.x_ss: featmat_ss, self.y: returns})

    def predict_many(self, paths):
        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            featmat_pix = np.concatenate([self._features_rgb(path) for path in paths])
            featmat_ss = np.concatenate([self._features(path) for path in paths])

        ret = self.session.run(self.test_net, {self.x_pix: featmat_pix,self.x_ss: featmat_ss})
        ret = np.reshape(ret, (ret.shape[0], ))
        return ret

    def predict(self, path):

        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            ret = self.session.run(self.test_net, {self.x_pix: self._features_rgb(path),self.x_ss: self._features(path)})
            ret = np.reshape(ret, (ret.shape[0], ))
            return ret

class VF_ss(object):
    coeffs = None

    def __init__(self, config, session):
        self.net = None
        self.config = config
        self.session = session
        # use exponential average when computing baseline
        self.averager = tf.train.ExponentialMovingAverage(decay=self.config.moving_average_vf)

    def init_vf(self,paths):
        if self.config.use_pixels:
            featmat = np.concatenate([self._features_rgb(path) for path in paths])
            return self.create_net(featmat.shape[1:])
        else:
            featmat = np.concatenate([self._features(path) for path in paths])
            return self.create_net([featmat.shape[1]])

    def fc_net(self, x, weight_loss_dict=None, reuse=None):
        net = x
        hidden_sizes = [64,64]
        for i in range(len(hidden_sizes)):
            net = linear(net, hidden_sizes[i], "vf/l{}".format(i), initializer=normalized_columns_initializer(1.0), weight_loss_dict=weight_loss_dict, reuse=reuse)
            net = tf.nn.elu(net)

        net = linear(net, 1, "vf/value", initializer=None, weight_loss_dict=weight_loss_dict, reuse=reuse)
        net = tf.reshape(net, (-1, ))
        return net, weight_loss_dict

    def conv_net42(self, x, weight_loss_dict=None, reuse=None):

        # Conv Layers
        for i in range(2):
            x = tf.nn.elu(conv2d(x, 32, "vf/l{}".format(i), [3, 3], [2, 2], \
                initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))

        x = flatten(x)
        # One more linear layer
        x = linear(x, 256, "vf/l{}".format(i+1), \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.nn.elu(x)

        x = linear(x, 1, "vf/value", \
            initializer=ortho_init(1), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.reshape(x, (-1, ))

        return x, weight_loss_dict

    def conv_net(self, x, weight_loss_dict=None, reuse=None):

        # Conv Layers
        x = tf.nn.relu(conv2d(x, 32, "vf/l0", [8, 8], [4, 4],pad="VALID", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))
        x = tf.nn.relu(conv2d(x, 64, "vf/l1", [4, 4], [2, 2],pad="VALID", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))
        x = tf.nn.relu(conv2d(x, 32, "vf/l2", [3, 3], [1, 1],pad="VALID", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse))

        x = flatten(x)
        # One more linear layer
        x = linear(x, 512, "vf/l3", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.nn.elu(x)

        x = linear(x, 1, "vf/value", \
            initializer=ortho_init(1), weight_loss_dict=weight_loss_dict, reuse=reuse)
        x = tf.reshape(x, (-1, ))

        return x, weight_loss_dict


    def create_net(self, shape):
        self.x = tf.placeholder(tf.float32, shape=[None] + list(shape), name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        self.vf_weight_loss_dict = {}
        with tf.name_scope('train_vf'):
            if self.config.use_pixels:
                self.net, self.vf_weight_loss_dict = self.conv_net(self.x, self.vf_weight_loss_dict)
            else:
                self.net, self.vf_weight_loss_dict = self.fc_net(self.x, self.vf_weight_loss_dict)

        self.bellman_error = (self.net - self.y)
        l2 = tf.reduce_mean(self.bellman_error * self.bellman_error)
        # get weight decay losses for value function
        vf_losses = tf.get_collection('vf_losses', None)

        self.loss = loss = l2 + tf.add_n(vf_losses)

        var_list_all = tf.trainable_variables()
        self.var_list = var_list = []
        for var in var_list_all:
            if "vf" in str(var.name):
                var_list.append(var)

        self.update_averages = self.averager.apply(self.var_list)

        # build test net with exponential moving averages for inference
        with tf.name_scope('test_vf'):
            if self.config.use_pixels:
                self.test_net, _ = self.conv_net(self.x, None, reuse=True)
            else:
                self.test_net, _ = self.fc_net(self.x, None, reuse=True)

        if self.config.use_adam_vf:
            self.loss_fisher = None
        else:
            sample_net = self.net + tf.random_normal(tf.shape(self.net))
            self.loss_fisher = loss_fisher = tf.reduce_mean(tf.pow(self.net - tf.stop_gradient(sample_net), 2))

        return self.loss, self.loss_fisher, self.vf_weight_loss_dict

    def init_vf_train_op(self, loss_vf, loss_vf_sampled, wd_dict):
        if self.config.use_adam_vf:
            # 0.001
            self.update_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_vf)
            self.queue_runner = None
        elif self.config.use_sgd_vf:
            # 0.001*(1.-0.9), 0.9
            self.update_op = tf.train.MomentumOptimizer(0.001*(1.-0.9), 0.9).minimize(loss_vf)
            self.queue_runner = None
        else:
            self.update_op, self.queue_runner = kfac.KfacOptimizer(
                                             learning_rate=self.config.lr_vf,
                                             cold_lr=self.config.lr_vf/3.,
                                             momentum=self.config.mom_vf,
                                             clip_kl=self.config.kl_desired_vf,
                                             upper_bound_kl=False,
                                             epsilon=self.config.epsilon_vf,
                                             stats_decay=self.config.stats_decay_vf,
                                             async=self.config.async_kfac,
                                             kfac_update=self.config.kfac_update_vf,
                                             cold_iter=self.config.cold_iter_vf,
                                             weight_decay_dict=wd_dict).minimize(
                                                  loss_vf,
                                                  loss_vf_sampled,
                                                  self.var_list)

        with tf.control_dependencies([self.update_op]):
            self.train = tf.group(self.update_averages)

        return self.train, self.queue_runner

    def _features(self, path):
        o = path["obs_ss"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        act = path["action_dists"].astype('float32')
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 10.0
        ret = np.concatenate([o, act, al, np.ones((l, 1))], axis=1)
        return ret

    def get_feed_dict(self, paths):
        if self.config.use_pixels:
            featmat = np.concatenate([self._features_rgb(path) for path in paths])
        else:
            featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        return {self.x: featmat, self.y: returns}

    def fit(self, paths):
        if self.config.use_pixels:
            featmat = np.concatenate([self._features_rgb(path) for path in paths])
        else:
            featmat = np.concatenate([self._features(path) for path in paths])
        if self.net is None:
            self.create_net(featmat.shape[1:])
        returns = np.concatenate([path["returns"] for path in paths])

        self.session.run(self.train, {self.x: featmat, self.y: returns})

    def predict_many(self, paths):
        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            if self.config.use_pixels:
                featmat = np.concatenate([self._features_rgb(path) for path in paths])
            else:
                featmat = np.concatenate([self._features(path) for path in paths])

        ret = self.session.run(self.test_net, {self.x: featmat})
        ret = np.reshape(ret, (ret.shape[0], ))
        return ret

    def predict(self, path):
        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            if self.config.use_pixels:
                ret = self.session.run(self.test_net, {self.x: self._features_rgb(path)})
            else:
                ret = self.session.run(self.test_net, {self.x: self._features(path)})
            ret = np.reshape(ret, (ret.shape[0], ))
            return ret

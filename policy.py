from utils import *
import utils

# universe-starter-agent 42x42 net
def create_policy_net_rgb42(obs, action_size):
    x = obs
    weight_loss_dict = {}

    # Conv Layers
    for i in range(2):
        x = tf.nn.relu(conv2d(x, 32, "policy/l{}".format(i), [3, 3], [2, 2], \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict))

    x = flatten(x)
    # One more linear layer
    x = linear(x, 256, "policy/l{}".format(i+1), \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict)
    x = tf.nn.relu(x)

    mean = linear(x, action_size, "policy/mean", ortho_init(1), weight_loss_dict=weight_loss_dict)
    log_std = tf.Variable(tf.zeros([action_size]), name="policy/log_std")
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output, weight_loss_dict

# universe-starter-agent 84x84 net
def create_policy_net_rgb84(obs, action_size):
    x = obs
    weight_loss_dict = {}

    # Conv Layers
    """for i in range(2):
        x = tf.nn.relu(conv2d(x, 32, "policy/l{}".format(i), [3, 3], [2, 2], \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict))
    """

    x = tf.nn.relu(conv2d(x, 32, "policy/l0", [8, 8], [4, 4],pad="VALID", \
        initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict))
    x = tf.nn.relu(conv2d(x, 32, "policy/l1", [4, 4], [2, 2],pad="VALID", \
        initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict))
    x = tf.nn.relu(conv2d(x, 32, "policy/l2", [3, 3], [1, 1],pad="VALID", \
        initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict))


    x = flatten(x)
    # One more linear layer
    x = linear(x, 256, "policy/l3", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict)
    x = tf.nn.relu(x)

    mean = linear(x, action_size, "policy/mean", ortho_init(1), weight_loss_dict=weight_loss_dict)
    log_std = tf.Variable(tf.zeros([action_size]), name="policy/log_std")
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output, weight_loss_dict

# universe-starter-agent 84x84 net
def create_policy_net_rgb63(obs, action_size):
    x = obs
    weight_loss_dict = {}

    # Conv Layers
    """for i in range(2):
        x = tf.nn.relu(conv2d(x, 32, "policy/l{}".format(i), [3, 3], [2, 2], \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict))
    """

    x = tf.nn.relu(conv2d(x, 32, "policy/l0", [3, 3], [2, 2],pad="VALID", \
        initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict))
    x = tf.nn.relu(conv2d(x, 32, "policy/l1", [3, 3], [2, 2],pad="VALID", \
        initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict))
    x = tf.nn.relu(conv2d(x, 32, "policy/l2", [3, 3], [2, 2],pad="VALID", \
        initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict))


    x = flatten(x)
    # One more linear layer
    x = linear(x, 256, "policy/l3", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict)
    x = tf.nn.relu(x)

    mean = linear(x, action_size, "policy/mean", ortho_init(1), weight_loss_dict=weight_loss_dict)
    log_std = tf.Variable(tf.zeros([action_size]), name="policy/log_std")
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output, weight_loss_dict

def create_policy_net_combi42(obs_pix, obs_ss, hidden_sizes, nonlinear, action_size):
    x_pix = obs_pix
    x_ss = obs_ss
    weight_loss_dict = {}

    # Conv Layers
    for i in range(2):
        x_pix = tf.nn.relu(conv2d(x_pix, 32, "policy/l{}".format(i), [3, 3], [2, 2], \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict))

    x_pix = flatten(x_pix)

    #  Linear Layers
    for i in range(len(hidden_sizes)):
        x_ss = linear(x_ss, hidden_sizes[i], "policy/l{}".format(i+2), initializer=normalized_columns_initializer(1.0), weight_loss_dict=weight_loss_dict)
        if nonlinear[i]:
            x_ss = tf.nn.tanh(x_ss)

    x_pix = linear(x_pix, 256, "policy/l4", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict)
    x_pix = tf.nn.relu(x_pix)


    x = tf.concat(1,[x_pix, x_ss])
    x = linear(x, 128, "policy/l5", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict)
    x = tf.nn.tanh(x)
    mean = linear(x, action_size, "policy/mean", ortho_init(1), weight_loss_dict=weight_loss_dict)
    log_std = tf.Variable(tf.zeros([action_size]), name="policy/log_std")
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output, weight_loss_dict

def create_policy_net_combi45(obs_pix, obs_ss, hidden_sizes, nonlinear, action_size):
    x_pix = obs_pix
    x_ss = obs_ss
    weight_loss_dict = {}

    # Conv Layers
    for i in range(3):
        x_pix = tf.nn.relu(conv2d(x_pix, 32, "policy/l{}".format(i), [3, 3], [2, 2],pad="VALID", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict))

    x_pix = flatten(x_pix)

    #  Linear Layers
    for i in range(len(hidden_sizes)):
        x_ss = linear(x_ss, hidden_sizes[i], "policy/l{}".format(i+3), initializer=normalized_columns_initializer(1.0), weight_loss_dict=weight_loss_dict)
        if nonlinear[i]:
            x_ss = tf.nn.tanh(x_ss)

    x = tf.concat(1,[x_pix, x_ss])

    x = linear(x, 256, "policy/l5", \
            initializer=ortho_init(np.sqrt(2)), weight_loss_dict=weight_loss_dict)
    x = tf.nn.relu(x)

    mean = linear(x, action_size, "policy/mean", ortho_init(1), weight_loss_dict=weight_loss_dict)
    log_std = tf.Variable(tf.zeros([action_size]), name="policy/log_std")
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output, weight_loss_dict

def load_policy_net_rgb(obs, policy_vars, action_size):
    x = obs
    intermediate_obs = []
    # Conv Layers
    for i in range(2):
        x = tf.nn.relu(conv2d_loaded(x, policy_vars[2*i], policy_vars[2*i+1], 32, [3,3], [2,2]))
        intermediate_obs.append(x)
    i+=1
    x = flatten(x)
    x = tf.nn.bias_add(tf.matmul(x, policy_vars[2*i]), policy_vars[2*i+1])
    x = tf.nn.relu(x)
    i += 1
    # Linear layer
    mean = tf.nn.bias_add(tf.matmul(x, policy_vars[2*i]), policy_vars[2*i+1])
    log_std = policy_vars[-1]
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output, intermediate_obs

def load_policy_net_ss(obs_ss, policy_vars, hidden_sizes, nonlinear, action_size):
    x = obs_ss
    for i in range(len(hidden_sizes)):
        x = tf.nn.bias_add(tf.matmul(x, policy_vars[2*i]), policy_vars[2*i+1])
        if nonlinear[i]:
            x = tf.nn.tanh(x)
    i+=1
    mean = tf.nn.bias_add(tf.matmul(x, policy_vars[2*i]), policy_vars[2*i+1])
    log_std = policy_vars[-1]
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output

def load_policy_net_combi42(obs_pix, obs_ss, policy_vars, hidden_sizes, nonlinear, action_size):
    x_pix = obs_pix
    x_ss = obs_ss

    # Conv Layers
    for i in range(2):
        x_pix = tf.nn.relu(conv2d_loaded(x_pix, policy_vars[2*i], policy_vars[2*i+1], 32, [3,3], [2,2]))
    i+=1
    x_pix = flatten(x_pix)

    #  Linear Layers
    for j in range(len(hidden_sizes)):
        x_ss = tf.nn.bias_add(tf.matmul(x_ss, policy_vars[2*i]), policy_vars[2*i+1])
        if nonlinear[j]:
            x_ss = tf.nn.tanh(x_ss)
        i+=1
    x_pix = tf.nn.bias_add(tf.matmul(x_pix, policy_vars[2*i]), policy_vars[2*i+1])
    x_pix = tf.nn.relu(x_pix)
    i+=1

    x = tf.concat(1,[x_pix, x_ss])
    x = tf.nn.bias_add(tf.matmul(x, policy_vars[2*i]), policy_vars[2*i+1])
    x = tf.nn.tanh(x)
    i += 1
    mean = tf.nn.bias_add(tf.matmul(x, policy_vars[2*i]), policy_vars[2*i+1])
    log_std = policy_vars[-1]
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output

def create_policy_net_ss(obs_ss, hidden_sizes, nonlinear, action_size):
    x = obs_ss
    weight_loss_dict = {}
    for i in range(len(hidden_sizes)):
        x = linear(x, hidden_sizes[i], "policy/l{}".format(i), initializer=normalized_columns_initializer(1.0), weight_loss_dict=weight_loss_dict)
        if nonlinear[i]:
            x = tf.nn.tanh(x)
    mean = linear(x, action_size, "policy/mean", initializer=normalized_columns_initializer(0.1), weight_loss_dict=weight_loss_dict)
    log_std = tf.Variable(tf.zeros([action_size]), name="policy/log_std")
    log_std_expand = tf.expand_dims(log_std, 0)
    std = tf.tile(tf.exp(log_std_expand), [tf.shape(mean)[0], 1])
    output = tf.concat(1, [tf.reshape(mean, [-1, action_size]), tf.reshape(std, [-1, action_size])])

    return output, weight_loss_dict

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os


import numpy as np
import tensorflow as tf
from dotmap import DotMap
import gym
import sys
sys.path.append(os.getcwd()+"/../..")

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC
import dmbrl.env



def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    if isinstance(goal_a, np.ndarray):
        return np.linalg.norm(goal_a - goal_b, axis=-1)
    return tf.norm(goal_a - goal_b, axis=-1)



class FetchPushConfigModule:
    ENV_NAME = "NongoalFetchPush-v0"
    TASK_HORIZON = 50
    NTRAIN_ITERS = 500
    NROLLOUTS_PER_ITER = 4
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 31 + 4, 31
    GP_NINDUCING_POINTS = 200



    def __init__(self):

        self.ENV = gym.make(self.ENV_NAME)
        self.ENV.seed(0)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 20,
                "max_iters": 5,
                "alpha": 0.1
            }

        }



    @staticmethod

    def obs_postproc(obs, pred):
        return obs + pred



    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs



    @staticmethod

    def obs_cost_fn(obs):
        achieved_goal = obs[:, -6:-3]
        goal = obs[:, -3:]

        # Compute distance between goal and the achieved goal.

        d = goal_distance(achieved_goal, goal)

        if isinstance(achieved_goal, np.ndarray):
            return (d > 0.05).astype(np.float32)
        return tf.cast(d > 0.05, tf.float32)



    @staticmethod

    def ac_cost_fn(acs):

        if isinstance(acs, np.ndarray):
            return 0.1 * np.sum(np.square(acs), axis=1)

        else:
            return 0.1 * tf.reduce_sum(tf.square(acs), axis=1)



    def nn_constructor(self, model_init_cfg):

        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(

            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),

            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),

            model_dir=model_init_cfg.get("model_dir", None)

        ))

        if not model_init_cfg.get("load_model", False):

            model.add(FC(200, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.000025))

            model.add(FC(200, activation="swish", weight_decay=0.00005))

            model.add(FC(200, activation="swish", weight_decay=0.000075))

            model.add(FC(200, activation="swish", weight_decay=0.000075))

            model.add(FC(self.MODEL_OUT, weight_decay=0.0001))

        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})

        return model



    def gp_constructor(self, model_init_cfg):

        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(

            name="model",

            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),

            kernel_args=model_init_cfg.get("kernel_args", {}),

            num_inducing_points=get_required_argument(

                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."

            ),

            sess=self.SESS

        ))

        return model





CONFIG_MODULE = FetchPushConfigModule

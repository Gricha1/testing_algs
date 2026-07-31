import os
import csv 
import pickle
import pathlib
import random

import jax
import jax.numpy as jnp
from jax import image

import embodied
import numpy as np
from gym import spaces
import pandas as pd
import cv2

from PIL import Image, ImageFont, ImageDraw
from craftax.craftax_env import make_craftax_env_from_name
from craftext.environment.craftext_wrapper import InstructionWrapper
from craftext.environment.craftext_wrapper_cmdp import CMDPInstructionWrapper


class Craftext(embodied.Env):

  def __init__(
    self,
    task,
    platform='gpu',
    mode="train",
  ):
    env_name="Craftax-Classic-Pixels-v1-Text"
    env_name = env_name.replace("-Text", "")
    env = make_craftax_env_from_name(env_name, False)
    #env = InstructionWrapper(env, craftext_settings)
    if task == "hungry":
      craftext_settings = "achievements_safe_budget_hungry"
    elif task == "drink":
      craftext_settings = "achievements_safe_budget_drink"
    elif task == "enegry":
      craftext_settings = "achievements_safe_budget_enegry"
    elif task == "hp":
      craftext_settings = "achievements_safe_budget_hp"
    else:
      assert 1 == 0, f"no task: {task}"
    
    env = CMDPInstructionWrapper(env, craftext_settings)
    self._env = env

    print("****************")
    print("env name:", env_name)
    print("craftext_settings:", craftext_settings)
    print("****************")

    from PIL import Image
    self._Image = Image
    from . import from_gymnasium
    self.wrappers = [
      from_gymnasium.FromGymnasium,
    ]

    #from . import from_gym
    #self.wrappers = [
    #  from_gym.FromGym,
    #]


    self.env_params = self._env.default_params
    observation_space_image = self._env.observation_space(self.env_params)
    action_space = self._env.action_space(self.env_params)

    image = spaces.Box(0, 255, (64, 64, 3), np.uint8)
    token_embed = spaces.Box(-np.inf, np.inf, shape=(768,), dtype=np.float16)
    is_read_step = spaces.Box(low=np.array(False), high=np.array(True), shape=(), dtype=bool)
    log_SR = spaces.Box(0, 1, shape=(), dtype=np.float16)
    obs_space = spaces.Dict({
          'image': image,
          'cost': spaces.Box(0, 1, shape=(), dtype=np.float16),
          'instruction_token_embed': token_embed,
          'constraint_token_embed': token_embed,
          "is_read_step": is_read_step,
          "log_success_rate": log_SR,
          "log_success_rate_reward": log_SR,
    })  
    
    self.observation_space = obs_space    
    self.action_space = action_space

  """
  def expand_array(self, array_):
    assert array_.dtype == np.float32
    array_ = jnp.asarray(array_, dtype=jnp.float32)  # Явное преобразование
    array_ = jnp.where(jnp.max(array_) > 1.0, array_ / 255.0, array_)
    if array_.shape == (130, 110, 3):
      expanded_array = image.resize(array_, (64, 64, 3), method='linear')
    elif array_.shape == (63, 63, 3):
      expanded_array = jnp.pad(array_, ((0, 1), (0, 1), (0, 0)), mode='constant', constant_values=0)
    else:
      raise ValueError(f"Input array must be of shape (63, 63, 3) but given {array_.shape}")
    
    return jnp.clip(expanded_array * 255, 0, 255).astype(jnp.uint8)
  """
  """
  @jax.jit
  def expand_array(self, array_):
      # Нормализация (как в оригинале)
      array_ = jnp.where(jnp.max(array_) > 1.0, array_ / 255.0, array_)
      
      # Обработка разных размеров входа
      def resize(x):
          if x.shape == (130, 110, 3):
              return jax.image.resize(x, (64, 64, 3), method='linear')
          elif x.shape == (63, 63, 3):
              padded = jnp.pad(x, ((0, 1), (0, 1), (0, 0)), mode='constant', constant_values=0)
              return jax.image.resize(padded, (64, 64, 3), method='linear')
          else:
              raise ValueError(f"Unsupported shape: {x.shape}")
      
      # Применяем resize и возвращаем в uint8 (как в оригинале)
      resized = resize(array_)
      return (jnp.clip(resized * 255, 0, 255)).astype(jnp.uint8)
  """

  
  def expand_array(self, array_):
    # Конвертируем в numpy array, если это еще не сделано
    if isinstance(array_, jax.Array):
        array_ = np.array(array_)
    
    # Нормализация значений
    array_ = np.where(np.max(array_) > 1.0, array_ / 255.0, array_)
    
    # Обработка разных размеров входа
    if array_.shape == (130, 110, 3):
        resized = cv2.resize(
            array_, 
            (64, 64), 
            interpolation=cv2.INTER_LINEAR
        )
    elif array_.shape == (63, 63, 3):
        # Добавляем паддинг перед ресайзом
        padded = np.pad(array_, ((0, 1), (0, 1), (0, 0)), mode='constant', constant_values=0)
        resized = cv2.resize(
            padded,
            (64, 64),
            interpolation=cv2.INTER_LINEAR
        )
    else:
        raise ValueError(f"Unsupported input shape: {array_.shape}")
    
    # Возвращаем в правильном формате
    return np.clip(resized * 255, 0, 255).astype(np.uint8)
  

  def reset(self):
    self.rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng = jax.random.split(self.rng)
    obs, self.env_state = self._env.reset(_rng, self.env_params)
    obs = self.expand_array(obs) # from 63 63 to 64 64

    dict_obs = {}
    dict_obs["image"] = obs
    dict_obs["instruction_token_embed"] = self.env_state.instruction.astype(np.float16)
    dict_obs["constraint_token_embed"] = self.env_state.textual_constraint.astype(np.float16)

    # fix
    dict_obs["cost"] = np.float32(0)
    dict_obs["log_success_rate"] = np.float32(0)
    dict_obs["log_success_rate_reward"] = float(0)

    dict_obs["is_read_step"] = False
    
    # additional check if task is finished
    self.reward_finish = False

    print("len task:", np.sum(self.env_state.instruction))

    return dict_obs, {}

  def step(self, action):
    self.rng, _rng = jax.random.split(self.rng)
    obs, self.env_state, reward, done, info = self._env.step(_rng, self.env_state, action, self.env_params)
    obs = self.expand_array(obs) # from 63 63 to 64 64
    cost = self.env_state.cost

    dict_obs = {}
    dict_obs["image"] = obs
    dict_obs["instruction_token_embed"] = self.env_state.instruction.astype(np.float16)
    dict_obs["constraint_token_embed"] = self.env_state.textual_constraint.astype(np.float16)
    dict_obs["cost"] = np.float32(cost)
    dict_obs["is_read_step"] = False
    dict_obs["log_success_rate"] = info["SR"]
  
    # additional check if task is finished
    if not self.reward_finish:
      if reward > 1:
        self.reward_finish = True
    dict_obs["log_success_rate_reward"] = float(self.reward_finish)

    return dict_obs, reward, cost, done, done, info
  
  def render(self):
    return self._env.render(mode="rgb_array")
  
  

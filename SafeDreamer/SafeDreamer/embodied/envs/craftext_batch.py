import os
import csv 
import pickle
import pathlib
import random
import functools

import jax
import jax.numpy as jnp
from jax import image

import embodied
import numpy as np
import gym
import pandas as pd
import cv2

from PIL import Image, ImageFont, ImageDraw
from craftax.craftax_env import make_craftax_env_from_name
from craftext.environment.craftext_wrapper import InstructionWrapper
from craftext.environment.craftext_wrapper_cmdp import CMDPInstructionWrapper
from craftext.wrappers.dreamer_wrappers_cmdp import (
    LogWrapper,
    DreamerOptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
)


class Craftext_Batch(embodied.Env):

  def __init__(
    self,
    task,
    platform='gpu',
    mode="train",
    num_envs=2,
    optimistic_reset_ratio=16,
    cost_koef=1.0,
  ):
    
    jax.config.update('jax_platform_name', 'gpu')
    jax.config.update('jax_transfer_guard', 'allow')

    env_name="Craftax-Classic-Pixels-v1-Text"
    env_name = env_name.replace("-Text", "")
    env = make_craftax_env_from_name(env_name, False)
    if task == "hungry":
      craftext_settings = "achievements_safe_budget_hungry"
    elif task == "drink":
      craftext_settings = "achievements_safe_budget_drink"
    elif task == "drinkeasy":
      craftext_settings = "achievements_easy_safe_budget_drink"
    elif task == "enegry":
      craftext_settings = "achievements_safe_budget_enegry"
    elif task == "hp":
      craftext_settings = "achievements_safe_budget_hp"
    else:
      assert 1 == 0, f"no task: {task}"
    
    self.num_envs = num_envs
    self.device = "gpu" # gpu
    with jax.default_device(jax.devices(self.device)[0]):
      env = CMDPInstructionWrapper(env, craftext_settings)
      env = LogWrapper(env)
      env = DreamerOptimisticResetVecEnvWrapper(
              env,
              num_envs=num_envs,
              reset_ratio=min(optimistic_reset_ratio, num_envs),
          )
      self._env = env
      self.cost_koef = cost_koef

      self._act_dict = hasattr(self._env.action_space, 'spaces')
      self._obs_dict = hasattr(self._env.observation_space, 'spaces')
      self._act_key = "action"
      #self._obs_key = "image"
      self.env_params = self._env.default_params

      self.rng = jax.device_put(jax.random.PRNGKey(42), jax.devices(self.device)[0])
      self.env_params = jax.device_put(self._env.default_params, jax.devices(self.device)[0])
      self._jit_step = jax.jit(self._raw_step, static_argnums=())
      self._jit_reset = jax.jit(self._raw_reset, static_argnums=())
      self.cumul_val_cost = 0
      self.cumul_val_reward = 0

      # initial reset
      _, self.env_state = self._jit_reset(self.rng, self.env_params)

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v()) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    print("act spaces:", spaces)
    return spaces
  
  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)
  

  @functools.cached_property
  def obs_space(self):
    image = gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
    image_original = gym.spaces.Box(0, 255, (130, 110, 3), np.uint8)
    token_embed = gym.spaces.Box(-np.inf, np.inf, shape=(768,), dtype=np.float16)
    is_read_step = gym.spaces.Box(low=np.array(False), high=np.array(True), shape=(), dtype=bool)
    obs_space = {
          'image': image,
          'instruction_token_embed': token_embed,
          'constraint_token_embed': token_embed,
          "is_read_step": is_read_step,
          "log_success_rate": embodied.Space(np.float32),
          "log_image_orignal": image_original,
    }
    spaces = self._flatten(obs_space)
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'cost': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  def __len__(self):
    return self.num_envs


  def _raw_step(self, rng, state, action, params):
    rng, step_rng = jax.random.split(rng)
    obs, new_state, reward, done, info = self._env.step(
        step_rng, state, action, params)
    return rng, new_state, obs, reward, done, info

  def _raw_reset(self, rng, params):
    rng, reset_rng = jax.random.split(rng)
    obs, state = self._env.reset(reset_rng, params)
    return obs, state
  
  def step(self, action):
    # Convert reset flags to jax array (NEW)
    reset_flags = jnp.array(action['reset'])
    
    # Prepare action dict for the wrapper (NEW)
    wrapper_action = {
        "reset": reset_flags,
        "action": jax.device_put(action[self._act_key], jax.devices(self.device)[0])
    }
    
    # Single JIT call (modified wrapper handles everything)
    self.rng, self.env_state, obs_origin, reward, done, info = self._jit_step(
        self.rng, self.env_state, wrapper_action, self.env_params
    )
    
    # Rest remains unchanged
    cost = self.env_state.env_state.cost * self.cost_koef
    
    if obs_origin.shape[-3:] == (63, 63, 3):
        obs = np.pad(obs_origin, ((0,0),(0,1),(0,1),(0,0)), mode='constant', constant_values=0)
        obs = (obs * 255).astype(np.uint8)  # Если значения в [0,1]
    elif obs_origin.shape[-3:] == (130, 110, 3):
        resized = np.zeros((obs_origin.shape[0], 64, 64, 3), dtype=np.uint8)
        for i in range(obs_origin.shape[0]):
            img = (obs_origin[i] * 255).astype(np.uint8) if obs_origin[i].max() <= 1.0 else obs_origin[i]
            resized[i] = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
        obs = resized
    else:
      assert 1 == 0

    is_first = action['reset']  # Reset envs are first
    is_last = done
    is_terminal = done

    dict_obs = {}
    dict_obs["image"] = obs
    dict_obs["instruction_token_embed"] = self.env_state.env_state.instruction.astype(np.float16)
    dict_obs["constraint_token_embed"] = self.env_state.env_state.textual_constraint.astype(np.float16)
    dict_obs["is_read_step"] = [False for _ in done]
    dict_obs["log_success_rate"] = [sr for sr in info["SR"]]
    if cost.shape == (1,): # validation
      if reset_flags.item():
        self.cumul_val_cost = 0
        self.cumul_val_reward = 0
      else:
        self.cumul_val_cost += cost
        self.cumul_val_reward += reward
        
      dict_obs["log_image_orignal"] = self.render_observation_image(obs, self.cumul_val_cost, self.cumul_val_reward)
    else:
      dict_obs["log_image_orignal"] = obs

    obs = self._obs(dict_obs, reward, cost, is_first, is_last, is_terminal)
    obs = {k: np.array(v) for k, v in obs.items()}

    return obs

  def _obs(
      self, obs, reward, cost, is_first=False, is_last=False, is_terminal=False):
    obs = self._flatten(obs)
    obs = {k: np.asarray(v) for k, v in obs.items()}
    obs.update(
        reward=np.float32(reward),
        cost=np.float32(cost),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs
  
  def render_observation_image(self, observation_image, cost, reward):
    # Создаем новое изображение с дополнительным местом для текста справа
    height, width = observation_image.shape[1:3]
    debug_width = 200  # Ширина области для debug информации
    total_width = width + debug_width
    
    # Функция для переноса текста
    def wrap_text(text, font, max_width):
        lines = []
        words = text.split()
        current_line = words[0]
        
        for word in words[1:]:
            test_line = current_line + " " + word
            # Получаем ширину текста в пикселях
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        lines.append(current_line)
        return lines
    
    # Создаем новое изображение с дополнительным пространством
    debug_images = []
    
    for i in range(observation_image.shape[0]):
        # Создаем основное изображение с debug панелью справа
        debug_image = np.zeros((height, total_width, 3), dtype=np.uint8)
        
        # Копируем оригинальное изображение в левую часть
        debug_image[:, :width, :] = observation_image[i]
        
        # Конвертируем в PIL для рисования текста
        pil_image = Image.fromarray(debug_image)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            # Пытаемся загрузить шрифт
            font = ImageFont.truetype("arial.ttf", 8)
        except:
            # Fallback на стандартный шрифт если arial недоступен
            font = ImageFont.load_default()
        
        # Получаем debug информацию
        cumul_cost = cost.item() if hasattr(cost, 'item') else float(cost)
        cumul_reward = reward.item() if hasattr(reward, 'item') else float(reward)
        
        # Получаем текстовые инструкции и ограничения
        idx = self.env_state.env_state.idx.item()
        instruction_embed = self._env.scenario_handler.scenario_data.instructions_list[idx]
        constraint_embed = self._env.scenario_handler.scenario_data.texutal_constraints_list[idx]
        
        # Максимальная ширина для текста (область справа минус отступы)
        max_text_width = debug_width - 20
        
        # Рисуем debug информацию в правой части
        x_position = width + 10
        y_position = 5
        
        # Cost и Reward
        draw.text((x_position, y_position), f"Cost: {cumul_cost:.2f}, koef {self.cost_koef}, limit={2}", font=font, fill=(255, 255, 255))
        y_position += 12
        draw.text((x_position, y_position), f"Reward: {cumul_reward:.2f}", font=font, fill=(255, 255, 255))
        y_position += 12
        
        # Instruction с переносом
        instruction_lines = wrap_text(f"Instr: {instruction_embed}", font, max_text_width)
        for line in instruction_lines:
            draw.text((x_position, y_position), line, font=font, fill=(255, 255, 255))
            y_position += 12
        
        # Constraint с переносом
        constraint_lines = wrap_text(f"Constr: {constraint_embed}", font, max_text_width)
        for line in constraint_lines:
            draw.text((x_position, y_position), line, font=font, fill=(255, 255, 255))
            y_position += 12
        
        # Добавляем разделительную линию
        draw.line([(width, 0), (width, height)], fill=(255, 255, 255), width=2)
        
        # Конвертируем обратно в numpy array
        debug_image_with_text = np.array(pil_image)
        debug_images.append(debug_image_with_text)
    
    return np.stack(debug_images)
  
  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result
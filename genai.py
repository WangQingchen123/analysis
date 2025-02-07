
import json  
from openai import OpenAI
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf




client = OpenAI(
    api_key="sk-piasulnonwqhjysuurglenfltayzarhrjkefxieeamnvrxpn", # 从https://cloud.siliconflow.cn/account/ak获取
    base_url="https://api.siliconflow.cn/v1"
)

response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V2.5",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": "? 2020 年世界奥运会乒乓球男子和女子单打冠军分别是谁? "
             "Please respond in the format {\"男子冠军\": ..., \"女子冠军\": ...}"}
        ],
        response_format={"type": "json_object"}
    )

print(response.choices[0].message.content)


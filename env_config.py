"""
环境配置 - 所有脚本共用
"""
import os
import warnings

# OpenMP冲突解决
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 线程控制
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# 警告控制
warnings.filterwarnings('ignore', category=UserWarning)

print("✓ 环境配置已加载")

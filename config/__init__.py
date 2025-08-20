"""Configuration module for BoardContinuity AI enterprise deployment"""

from .production import ProductionConfig, get_config, ENVIRONMENT_CONFIGS

__all__ = ['ProductionConfig', 'get_config', 'ENVIRONMENT_CONFIGS']
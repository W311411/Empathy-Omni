
from .speech_generator_ar_mtp import SpeechGeneratorARMTP
from .speech_generator import LLMSpeechGenerator

def build_speech_generator(config):
    generator_type = getattr(config, 'speech_generator_type')
    if 'ar_mtp' in generator_type:
        return SpeechGeneratorARMTP(config)
    if 'new' in generator_type:
        return LLMSpeechGenerator(config)
    raise ValueError(f'Unknown generator type: {generator_type}')

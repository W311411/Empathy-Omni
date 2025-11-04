from .speech_encoder import WhisperWrappedEncoder, Emotion2vecEncoder
def build_speech_encoder(config):
    speech_encoder_type = getattr(config, 'speech_encoder_type', None)
    if speech_encoder_type is None:
        raise ValueError('speech_encoder_type must be specified in the configuration.')
    if "whisper" == speech_encoder_type.lower():
        return WhisperWrappedEncoder.load(config)


    raise ValueError(f'Unknown speech encoder: {speech_encoder_type}')

def build_emotion_encoder(config):
    if "emotion2vec" in config.emotion_encoder.lower():
        emotion_encoder_type = 'emotion2vec'
        print("初始化emo2vec编码器")
        return Emotion2vecEncoder.load(config)


    raise ValueError(f'Unknown emotion encoder: {emotion_encoder_type}')



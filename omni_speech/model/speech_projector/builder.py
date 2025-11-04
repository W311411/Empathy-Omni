from .speech_projector import EncoderProjectorConcat, EmotionProjectorConcat, EmotionPredictHead, EmotionClassificationHead


def build_speech_projector(config):
    projector_type = getattr(config, 'speech_projector_type', 'linear')
    if projector_type == 'linear':
        return EncoderProjectorConcat(config)

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_emotion_projector(config):

    return EmotionProjectorConcat(config)

def build_emotion_predict_head(config):
    return EmotionPredictHead(config)

def build_emotion_classifier():
    return EmotionClassificationHead()

   

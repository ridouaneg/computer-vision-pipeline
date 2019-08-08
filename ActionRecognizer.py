import mxnet as mx
import cv2
import numpy as np
import time

class ActionRecognizer:

    def __init__(self, model_name='vgg16_ucf101', input_size=(224, 224, 3), threshold=0.20, do_timing=False):
        self.threshold = threshold
        self.input_size = input_size
        self.image_size = None
        self.do_timing = do_timing
        self.result = None

        self.set_model(model_name)

        self.classes = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', \
        'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', \
        'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', \
        'CricketBowling', 'CricketShot', 'CuttingInKitchen', 'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', \
        'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'HammerThrow', 'Hammering', 'HandstandPushups', 'HandstandWalking', \
        'HeadMassage', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpRope', \
        'JumpingJack', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor', 'Nunchucks', 'ParallelBars', \
        'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', \
        'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', \
        'RopeClimbing', 'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', \
        'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot', 'TaiChi', 'TennisSwing', 'ThrowDiscus', \
        'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard', 'YoYo']

    def set_model(self, model_name):
        from gluoncv.model_zoo import get_model
        self.model = get_model('vgg16_ucf101', nclass=101, ctx=mx.gpu(0))

    def get_result(self):
        return self.result

    def predict(self, image):

        # pre-process
        pre_process_runtime_start = time.time()
        model_input = self.pre_process(image)
        pre_process_runtime_end = time.time()

        # model prediction
        model_predict_runtime_start = time.time()
        model_output = self.model_predict(model_input)
        model_predict_runtime_end = time.time()

        # postprocess
        post_process_runtime_start = time.time()
        #actions, confidences = self.post_process(model_output)
        self.post_process(model_output)
        post_process_runtime_end = time.time()

        #self.result = ActionRecognizerResult([Action(keypoints, confidence) for keypoints, confidence in zip(poses, confidences)])

        if self.do_timing:
            print('action classifier preprocessing time (ms):', (pre_process_runtime_end - pre_process_runtime_start) * 1e3)
            print('action classifier prediction time (ms):', (model_predict_runtime_end - model_predict_runtime_start) * 1e3)
            print('action classifier post-processing time (ms):', (post_process_runtime_end - post_process_runtime_start) * 1e3)

    def pre_process(self, image):
        self.image_size = np.shape(image)

        model_input = cv2.resize(image, (self.input_size[1], self.input_size[0]))

        model_input = np.expand_dims(model_input, 0)
        model_input = np.swapaxes(model_input, 1, 3)
        model_input = np.swapaxes(model_input, 2, 3)

        model_input = model_input / 255.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        mean, std = np.reshape(mean, (1, 3, 1, 1)), np.reshape(std, (1, 3, 1, 1))
        model_input = (model_input - mean) / std

        model_input = mx.nd.array(model_input, ctx=mx.gpu(0))

        return model_input

    def model_predict(self, model_input):
        model_output = self.model(model_input)
        model_output = mx.nd.softmax(model_output)
        return model_output

    def post_process(self, model_output):
        model_output = model_output.asnumpy()
        probabilities = np.max(model_output)
        actions_idx = np.argmax(model_output)
        actions_text = self.classes[actions_idx]
        print(actions_text, probabilities)

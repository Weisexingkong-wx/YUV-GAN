

from config import FLAGS
import initializer as init
from cloud_gan import CloudRemovalGAN


FLAGS.batch_size = 1
params = init.TrainingParamInitialization()
nets = CloudRemovalGAN(params)
nets.run_test_loop()

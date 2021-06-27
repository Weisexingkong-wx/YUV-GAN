
import initializer as init
from cloud_gan import CloudRemovalGAN
# All parameters used in this file

if __name__ == '__main__':
    params = init.TrainingParamInitialization()
    nets = CloudRemovalGAN(params)
    nets.run_train_loop()
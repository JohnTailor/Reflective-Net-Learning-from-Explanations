# Reflective-Net-Learning-from-Explanations
# Code for Paper: Reflective-Net: Learning from Explanations
https://arxiv.org/abs/2011.13986


The code shows significant gains for reflective-net. It uses Python and Pytorch.

Run learnFExp.py and wait until a few runs are performed.

The code in this repo is simplified for better readability. There has been no hyperparameter tuning on this version.
Other layers might be used for explanation that further improve performance.
Also the parameter 'maxRan' is introduced. However, improvements are significant with/without using it (ie. also for maxRan==1)" )


For maxRun==1.0 (as in original paper): 12 runs yielded a gain due to reflective nets of about 0.63 +/- 0.35 for CIFAR-10 and 0.9 +/- 0.8 for CIFAR-100"
For maxRun==0.5: 12 runs yielded a gain due to reflective nets of about 0.3 +/- 0.2 for CIFAR-10 and 1.7 +/- 0.4 for CIFAR-100"




To load the models, consider the following.

MNIST_autoencoder uses the latest AutoEncoder architecture with d=48, k=6.

MNIST_classifier uses the latest Classifier architecture with d=16, k=6. It achieves 98% accuracy.

MNIST_vae_smalle uses d=32, k=6, n=1, sigma=0.2. It achieves MSE=10.4 and KL=48.8

MNIST_vae_medium uses d=64, k=6, n=3, sigma=0.2. It achieves MSE=9.5 and KL=41.8

MNIST_flowmathing uses the vae_medium space with d=32, k=6. It achieves mse=0.50. Conditioned images with w>=3 look good. Unconditioned images don't.

-----

MNIST_vae_small uses the (1, 3, 3) with d=48, k=5, n=2, sigma=0.2 (231K). It achieves mse=10.9, kl=34.4.

MNIST_vae_dim12_small uses d=32, k=6, n=0, sigma=0.2, z_dim=12 and got to mse=10.3, kl=36.9 with total_epochs = 45, milestones = [15, 20, 25, 30, 34, 37, 40], gamma = 0.8

MNIST_flowmatching_dim12_small is trained on the equivalent vae. Uses 2 hidden layers of size 150 and reaches mse=1.29. lr = 1e-2, total_epochs = 15, milestones = [5, 10], gamma = 0.1

there's also some for dim=15, sigma=0.1 but they are slightly worse

------

The arguments for FashionMNIST are in the training cells. The L1 version uses L1 loss instead of L2

flowmatching_new doesn't use an autoencoder, uses architecture 2 with d=32, k=5, n=2, n_heads=4, head_dim=6 (46kk). achieves mse=0.16 (lowest by far)

------

EMNIST_classifier_big uses d1=48, d2=64, k=6, n=2, n_heads=6, head_dim=8 (140k). Achieves 86% top1 and 95.5% top2

EMNIST_classifier uses d1=16, d2=24, k=5, n=1, n_heads=2, head_dim=8 (11k). Achieves 86.6% top1 and 95.7% top2

EMNIST_autoencoder_L1 uses d1=24, d2=32, k=4, n=1, n_heads=3, head_dim=8, p_norm=1, (41k). Achieves L1=18.2

EMNIST_flowmatching uses d=48, k=5, n=2, n_heads=5, head_dim=8, n_class=num_classes, p=0.2. Achieves mse=1.27

EMNIST_vae uses d1=48, d2=72, k=6, n=3, n_heads=5, head_dim=12, sigma=0.2 (572k). Achieves mse=10.23, kl=60.7
    Most variances are small, but there's a small second peak near 0.8

EMNIST_flowmatching_vae uses d1=24, d2=32, k=5, n=1, n_heads=3, head_dim=8, n_class=47, p=0.1, z_dim=6. Achieves mse=1.04

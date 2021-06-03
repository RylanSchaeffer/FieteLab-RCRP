echo -n password:
read -s password

#sshpass -v -p $password rsync -avh --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_00_crp_prior/plots/ exp_00_crp_prior/plots/
sshpass -v -p $password rsync -avh --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_01_mixture_of_gaussians/plots/ exp_01_mixture_of_gaussians/plots
sshpass -v -p $password rsync -avh --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_02_mixture_of_unigrams/plots/ exp_02_mixture_of_unigrams/plots/
sshpass -v -p $password rsync -avh --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_03_reddit/plots/ exp_03_reddit/plots/
sshpass -v -p $password rsync -avh --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_04_newsgroup/plots/ exp_04_newsgroup/plots/
sshpass -v -p $password rsync -avh --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_08_omniglot/plots/ exp_08_omniglot/plots/
echo -n password:
read -s password

#sshpass -v -p $password rsync -avh --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RCRP/exp_00_crp_prior/plots/ exp_00_crp_prior/plots/
sshpass -v -p $password rsync -avh --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RCRP/exp_01_mixture_of_gaussians/plots* exp_01_mixture_of_gaussians/
sshpass -v -p $password rsync -avh --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RCRP/exp_02_mixture_of_unigrams/plots* exp_02_mixture_of_unigrams/
#sshpass -v -p $password rsync -avh --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RCRP/exp_03_reddit/plots/ exp_03_reddit/plots/
sshpass -v -p $password rsync -avh --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RCRP/exp_04_newsgroup/plots* exp_04_newsgroup/
sshpass -v -p $password rsync -avh --exclude='*.joblib' rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-RCRP/exp_08_omniglot/plots* exp_08_omniglot/

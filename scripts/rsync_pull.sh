echo -n password:
read -s password

sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_00_crp_prior/plots/ exp_00_crp_prior/plots/
sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_01_mixture_of_gaussians/plots/ exp_01_mixture_of_gaussians/plots
sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_02_mixture_of_unigrams/plots/ exp_02_mixture_of_unigrams/plots/
#sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_03_language_modeling/plots/ exp_03_language_modeling/plots/
#sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_07_olfactory/ exp_07_olfactory/
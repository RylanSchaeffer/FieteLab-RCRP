echo -n Password:
read -s password

sshpass -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_01_mixture_of_gaussians/ exp_01_mixture_of_gaussians/
sshpass -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_02_dp_gmm/ exp_02_dp_gmm/
sshpass -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_04_mixture_of_unigrams/ exp_04_mixture_of_unigrams/
sshpass -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_05_language_modeling/ exp_05_language_modeling/
sshpass -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_06_ibp_prior/ exp_06_ibp_prior/
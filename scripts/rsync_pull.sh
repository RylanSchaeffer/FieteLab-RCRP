echo -n password:
read -s password

#sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_00_crp_prior/ exp_00_crp_prior/
sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_01_mixture_of_gaussians/ exp_01_mixture_of_gaussians/
sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_02_mixture_of_unigrams/ exp_02_mixture_of_unigrams/
#sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_05_language_modeling/ exp_05_language_modeling/
#sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_06_ibp_prior/ exp_06_ibp_prior/
#sshpass -v -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_07_olfactory/ exp_07_olfactory/
echo -n Password:
read -s password

sshpass -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_01_mixture_of_gaussians/plots/ exp_01_mixture_of_gaussians/plots/
sshpass -p $password rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_02_dp_gmm/plots/ exp_02_dp_gmm/plots/

#rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_01_mixture_of_gaussians/plots/ exp_01_mixture_of_gaussians/plots/
#rsync -avh rylansch@openmind7.mit.edu:/om2/user/rylansch/FieteLab-ddCRP/exp_02_dp_gmm/plots/ exp_02_dp_gmm/plots/
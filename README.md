# Semi-Markov Offline RL 

This codebase stores the code for a series of experiments to: 1) compare MDP- vs SMDP-based algorithms in a customized Minigrid 8x8 environment; and, 2) compare three SMDP-based algorithms in an offline learning environment.  

This codebase uses wandb. To set up wandb, please see https://docs.wandb.ai/quickstart for details. To use wandb for the experiments, update the "wandb_key" parameter in the config_minigrid.yaml file. 

## Online Experiments
To run a wandb sweep for the online experiments, 
begin by running the following.

    $ wandb sweep sweep.yaml

The output of the command will be the sweep ID, which is used in the following step.

    $ wandb agent <USERNAME/PROJECTNAME/SWEEPID>

The training progress can be monitored on the wandb dashboard. Once the sweep is complete, run the following command to generate the plots.

    $ python evaluation/plot_returns.py 

To run the online experiments without wandb, you will need to run train.py over the parameters specified in sweep.yaml. Each train.py file can 
be run individually as the following, using the different flags. For example, the following three lines 
train a BCQ network, a DQN network, and a DDQN network, respectively. 


    $ python train.py --network_size bcq_fc 
    $ python train.py --network_size dqn_fc 
    $ python train.py --network_size dqn_fc --ddqn 

## Offline Experiments
For the offline experiments, we need to create replay buffers using a combination of optimal and sub-optimal actions. 
The optimal actions are chosen using a policy learned in the online setting. 
The optimal policy is the one learned in the online setting. To specify which policy is the optimal policy, 
update the "buffer_name" key in the config_minigrid.yaml file and "model_name" in the buffer_config.yaml file. 
A sample config file "sample_config/config_minigrid_sample_offline.yaml" is provided.

Then, run the following line to set up a wandb sweep to generate the buffers. The output of this line is again the sweep ID,
which is used to run a sweep agent. 

    $ wandb sweep sweep_buffer.yaml 

Alternatively, each buffer can be created one at a time using the following script, which uses the parameters 
specified in buffer_config.yaml.

    $ python generate_buffer.py

Once the buffers are created, run the following script to learn policies for each buffer.
If wandb is not used, each train.py file can be run with the parameters specified in config_minigrid.yaml.


    $ wandb sweep sweep_offline.yaml

Once the policies are learned using the different buffers, we can plot the results using the following script.

    $ python evaluation/plot_buffer_analysis.py 

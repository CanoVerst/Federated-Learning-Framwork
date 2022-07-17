from audioop import avg
from platform import python_branch
from shutil import which
from unittest import result
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

def plot(csv_to_plot, csv_std_min, csv_std_max, exp_names, save_file):
    #We start plotting here
    TSBOARD_SMOOTHING = 0.6
    INTERVAL = 2.0 #interval for how much you want to space out the graph
    
    csv = csv_to_plot

    sns.color_palette("flare", as_cmap=True)
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")
    f, ax1 = plt.subplots(figsize=(8.5,5))

    csv = csv.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
    csv_std_min = csv_std_min.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
    csv_std_max = csv_std_max.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
    
    for i in range(len(csv.columns)):
        #get csv index
        rewards = csv[i]
        ax1 = sns.lineplot(data=rewards, linewidth=2.5, label= exp_names[i])
        #ax1.fill_between(rewards.index, csv_std_max[i].values, csv_std_min[i].values, alpha=0.5)

    ax1.set_xlabel("Training round", fontstyle='normal')
    ax1.set_ylabel("Cumulative average reward")
    ax1.xaxis.set_major_formatter(plticker.FormatStrFormatter('%d'))
    loc = plticker.MultipleLocator(base=INTERVAL) # this locator puts ticks at regular intervals of two
    ax1.xaxis.set_major_locator(loc)
    plt.xlim(0, len(csv)-1)

    f.savefig(f'{save_file}.pdf', bbox_inches='tight')

def combine_n_dataframes(csv_list, csv_min_list, csv_max_list, num_culsum_columns):
    csv = pd.concat(csv_list, axis=1, join='inner')
    csv_min = pd.concat(csv_min_list, axis=1, join='inner')
    csv_max = pd.concat(csv_max_list, axis=1, join='inner')

    for i in range(num_culsum_columns):
        if i == 0:
            continue
        csv.columns.values[i] = i
        csv_min.columns.values[i] = i
        csv_max.columns.values[i] = i

    return csv, csv_min, csv_max
    
def generate_culsum_avg(seeds, experiment, target_file, num_columns):
    """Comments use 3 seeds as the example but extends up to n seeds"""
    eval_all_file = f'{experiment}_ALL'
    names_of_columns = []
    num_seeds = len(seeds)

    for i in range(num_columns):
        names_of_columns.append(i)
    
    rewards = pd.read_csv(f'{eval_all_file}.csv', names=names_of_columns)
    print(eval_all_file)

    for i in range(num_seeds):
        #Generating the sumlist
        sumlist = []
        curr_seed_count = 0
        while curr_seed_count < num_seeds:
            sumlist.append(i*num_seeds+curr_seed_count)
            curr_seed_count += 1
        #Calculate the sum down the rows of the columns we want
        rewards[i] = rewards[sumlist].sum(axis=1)

    #Only want the first 3 columns 
    rewards = rewards.iloc[:, 0:num_seeds]

    #Get the std dev of the 3 columns
    rewards_std_dev = rewards.std(axis=1)

    #Take the mean of the 3 columns along the rows
    rewards = rewards.mean(axis=1)

    rewards_min = rewards - rewards_std_dev
    rewards_max = rewards + rewards_std_dev

    rewards = pd.DataFrame(rewards)
    rewards_min = pd.DataFrame(rewards_min)
    rewards_max = pd.DataFrame(rewards_max)

    print(target_file)
    rewards.to_csv(f'{target_file}.csv', index=False, header=None)

    return rewards, rewards_min, rewards_max

if __name__ == '__main__':
    seeds = [5,10,15]
    NUM_TASKS = 3

    """Make sure that experiments at index i match up with target files at index i
    For example, percentile aggreagte fisher at index 0 for experiments should be the same for
    target_files at index 0 (both fisher). Make sure exp_names list matches too (at index 0 it is 
    also fisher)."""

    """To make it support more experiments, seeds, and tasks all you would need to do is change the
    number of tasks and the seeds, experiments, and target_files list"""

    experiments = ["A2C_RL_SERVER_PERCENTILE_AGGREGATE_LAMDA5", "A2C_RL_SERVER_FED_AVG", "A2C_RL_SERVER_PERCENTILE_ACTOR", "A2C_RL_SERVER_PERCENTILE_CRITIC", "A2C_RL_SERVER_PERCENTILE_ACTOR_LOSS", "A2C_RL_SERVER_PERCENTILE_ACTOR_GRAD", "A2C_RL_SERVER_PERCENTILE_CRITIC_LOSS", "A2C_RL_SERVER_PERCENTILE_CRITIC_GRAD"]
    target_files = ["FISHER_CULSUM_AVG_LAMDA5", "FEDAVG_CULSUM_AVG", "FISHER_CULSUM_AVG_ACTOR", "FISHER_CULSUM_AVG_CRITIC", "FISHER_CULSUM_AVG_ACTOR_LOSS", "FISHER_CULSUM_AVG_ACTOR_GRAD", "FISHER_CULSUM_AVG_CRITIC_LOSS", "FISHER_CULSUM_AVG_CRITIC_GRAD"]
    exp_names = ["Curriculum + MAS", "FedAvg", "Curriculum actor", "Curriculum critic", "Curriculum actor loss", "Curriculum actor grad", "Curriculum critic loss", "Curriculum critic grad"]

    #number of columns for the culmunating sum final csv file
    num_culsum_columns = len(experiments)

    #number of columns from the original csv file we are reading from
    num_original_columns = len(seeds) * NUM_TASKS

    avg_rewards_of_experiments = []
    #The minimum and maximium bound of the experiments
    min_bound= []
    max_bound = []

    for i in  range(len(experiments)):
        rewards, rewards_min, rewards_max = generate_culsum_avg(seeds, experiments[i], target_files[i], num_original_columns)
        avg_rewards_of_experiments.append(rewards)
        min_bound.append(rewards_min)
        max_bound.append(rewards_max)

    save_file = "culsum_fedavg_MAS_actor_actorloss_actorgrad_critic_criticloss_criticgrad_"
    plot_csv, plot_min, plot_max = combine_n_dataframes(avg_rewards_of_experiments, min_bound, max_bound, num_culsum_columns)
    print(plot_csv)
    plot(plot_csv, plot_min, plot_max, exp_names, save_file)

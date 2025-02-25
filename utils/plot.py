import matplotlib.pyplot as plt
import config

def plot_losses(plots ,Xlabel ,Ylabel, title_name):

    plt.figure(figsize=(12, 5))
    for key, value in plots.items():
        losses = value
        plt.plot(range(1, len(losses) + 1), value, label=key)

    plt.xlabel(f'{Xlabel}')
    plt.ylabel(f'{Ylabel}')
    plt.title(f"{title_name}")
    plt.legend()
    plt.grid(True)

    plt.savefig(f'Losses_global_{config.NUM_EPOCHS}.png')
    plt.close()

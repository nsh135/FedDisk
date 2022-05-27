import pickle 
import os, sys
import matplotlib.pyplot as plt


def plot_loss(dir):
    pkl_file = dir +'/clients.pkl'
    with open(pkl_file,'rb') as f:
        clients = pickle.load(f)

    # get path
    figure_path = dir+'/Figures/'

    ## plot training loss in different clients
    plt.figure()
    for client in clients:
        list_history = client['histories']
        loss = [d['loss'][0] for d in list_history]
        plt.plot(loss, label= '{}'.format(client['dataset']))
        plt.legend()
        plt.xlabel('Global Interations')
        plt.ylabel('Training Loss')
    print("** Loss plot is saved under {}".format(figure_path+'loss.jpg'))
    plt.savefig(figure_path+'loss.jpg')
    plt.close()


if __name__ == '__main__':
    """ """
    assert len(sys.argv)<3, "Please provide log directory as the argument,\n E.g., python result_plot.py 25012022_16:00:18" 
    dir = '../Logs/'+sys.argv[1]
    plot_loss(dir)
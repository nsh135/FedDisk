""" Main file to train and test federated learning """

from gc import callbacks
from itertools import combinations
from tokenize import String
import  tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np 
import random as rn
import argparse
import pickle, os
from models import benchmark_models, CVAE, vae_training, toGrey
from skimage.transform import resize
from PIL import Image
from datetime import datetime
from utilities import *
from sklearn.linear_model import LogisticRegression
from result_plot import plot_loss
from sklearn.model_selection import train_test_split
from MADE import made_model,display_digits

# for reproducing hashbased algorigsm 
os.environ['PYTHONHASHSEED'] = '0'

datasets = [ 'SynthDigits', 'MNIST_M', 'SVHN','MNIST', 'USPS', ]
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
n_classes = 10


# prepare Logs directory 
now = datetime.now()
dt = now.strftime("%d%m%Y_%H:%M:%S")
log_dir = '../Logs/{}'.format(dt)
logfile = log_dir + '/'+ 'log.log'
figures_dir= log_dir + '/Figures'
models_dir = log_dir + '/Models'

if not os.path.exists(log_dir):
  os.makedirs(log_dir)
if not os.path.exists(log_dir):
  os.makedirs(models_dir)
if not os.path.exists(figures_dir):
  os.makedirs(figures_dir+'/Samples')


def prepare_data(dataset):
    """
    dataset names: MNIST, MNIST_M, SVHN, SynthDigits, USPS
    """
    for p in range(10):   
        file = open('../Data/{}/partitions/train_part{}.pkl'.format(dataset, p), 'rb')
        X, y = pickle.load(file)
        file.close()
        if p==0: # first partition
            X_train=X
            y_train=y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)
    #load testing file 
    file = open('../Data/{}/test.pkl'.format(dataset), 'rb')
    X_test, y_test = pickle.load(file)
    file.close()
    
    #resize images and scale (/255)
    X_train = np.array(resize(X_train, (X_train.shape[0],28,28,3)), dtype= 'float32')
    X_test = np.array(resize(X_test, (X_test.shape[0],28,28,3)), dtype= 'float32')

    #convert to grey
    
    if toGrey:
        X_train  = np.mean(X_train,axis=-1).reshape((-1,28,28,1))
        X_test = np.mean(X_test,axis=-1).reshape((-1,28,28,1))
    else:
        # save some samples for sanity check
        x_recover = np.rint(X_train[0]*255).astype(np.uint8)
        pil_image1=Image.fromarray(x_recover, mode='RGB')
        pil_image1.save(figures_dir + '/Samples/img_after_resized_{}_{}.jpg'.format(dataset, y_train[0]))

    # permutation 
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.0001, random_state=42)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.0001, random_state=42)
    X_test = X_test[:1860]; y_test = y_test[:1860]

    print("Loading Dataset: {}".format(dataset))
    print("X_train.shape: {}".format( X_train.shape) )
    print("y_train.shape: {}".format( y_train.shape) )
    print("X_test.shape: {}".format(X_test.shape))
    print("y_test.shape: {}".format( y_test.shape))
    return X_train, y_train, X_test, y_test


def normalize(x):
    """ normalize array x to [0,1]"""
    r = (x-np.min(x))/(np.max(x) - np.min(x))*2
    print("******r:{} \nmean r:{} std r:{}".format(x, np.mean(r) , np.std(r) ))
    return r


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default='fedisk',choices= ['fedbn','fedavg','fedisk'],  help = 'method to use, ')
    parser.add_argument('--seed', type = int, default= 13, help ='seed')
    parser.add_argument('--n_trials', type = int, default= 3, help ='number of trials')
    parser.add_argument('--params_search', type = bool, default= False, help ='test  with the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 128, help ='batch size')
    parser.add_argument('--n_clients', type = int, default= 3, help ='number of clients between 2-5')
    parser.add_argument('--iters', type = int, default=50, help = 'global training iterations')
    parser.add_argument('--vae_iters', type = int, default=100, help = 'vae iterations')
    parser.add_argument('--vae_latent_dim', type = int, default=10, help = 'vae latent dimention')
    parser.add_argument('--vae_batch', type = int, default= 128, help ='batch size for training vae')
    parser.add_argument('--made_iters', type = int, default=100, help = 'MADE iterations')
    parser.add_argument('--made_hidden_neurons', type = int, default=1000, help = 'number of neuron per layer for MADE model')
    parser.add_argument('--made_hidden_layers', type = int, default=3, help = 'number of hidden layers in MADE model')
    parser.add_argument('--made_batch', type = int, default= 128, help ='batch size for training vae')
    parser.add_argument('--logistic_iter', type = int, default=50, help = 'max logistic regression iterations')
    parser.add_argument('--shallow_neuron', type = int, default= 200, help ='number of neuron for shallow network(calculating weight)')
    parser.add_argument('--shallow_iters', type = int, default= 40, help ='number of itersation for shallow network')
    parser.add_argument('--note',  default= '', help ='A note will be written down in the log')
    parser.add_argument('--correction_method',  default= 'density_weight', choices= ['data_weight','density_weight'])
    parser.add_argument('--model_path',  default='', help = 'an existing log dir where models were stored ')

    global args
    args = parser.parse_args()

    # Set random seed for reproducing
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rn.seed(args.seed)

    if args.params_search:
        made_iters_list =  [500]
        made_hidden_neurons_list = [1000, 5000]
        made_hidden_layers_list = [50, 100]
        made_batch_list = [128]
        shallow_iters_list = [20, 40 ] 
        ## grid search parameters
        params = []
        for made_it in made_iters_list:
            for neurons in made_hidden_neurons_list:
                for layers in made_hidden_layers_list:
                    for m_batch in made_batch_list:
                        for shallow_it in shallow_iters_list:
                            params.append( (made_it,neurons,layers,m_batch,shallow_it ) )
        print("Number of Search Params :{}".format(len(params)))
    else:
        params = [ (args.made_iters, args.made_hidden_neurons, args.made_hidden_layers, args.made_batch, args.shallow_iters) ]



    final_results= []
    for param in params:
        made_iters, made_hidden_neurons, made_hidden_layers, made_batch, shallow_iters = param
        #store trial results
        trial_results= {}
        for i in range(args.n_clients):
            trial_results[datasets[i]] = []

        for trial in range(args.n_trials):
            print("\n----------TRIAL {}/{}-----------".format(trial,args.n_trials))
            # prepare data and Clients dictionary to store variables
            clients=[] ## list of clients , each contains a dict of data and model
            for i in range(args.n_clients):
                X_train, y_train, X_test, y_test = prepare_data(dataset=datasets[i])
                clients.append({'dataset':datasets[i], 'X_train': X_train, 'y_train': y_train, \
                    'X_test': X_test, 'y_test': y_test ,'histories':[], 'acc':[],\
                    'sample_weights' : None  } )
                clients[i]['vae_train_dataset'] = (tf.data.Dataset.from_tensor_slices(X_train)
                        .shuffle(X_train.shape[0]).batch(args.vae_batch))
                clients[i]['vae_test_dataset'] = (tf.data.Dataset.from_tensor_slices(X_test)
                        .shuffle(X_test.shape[0]).batch(args.vae_batch))
                #for density made model
                if not toGrey: # dim = 28x28x3
                    clients[i]['made_train_dataset'] = np.mean(X_train, axis=-1).reshape((X_train.shape[0],-1) )
                    clients[i]['made_test_dataset'] = np.mean(X_test, axis=-1).reshape((X_test.shape[0],-1) )
                else:# dim = 28x28x1
                    clients[i]['made_train_dataset'] = X_train.reshape((X_train.shape[0],-1) )
                    clients[i]['made_test_dataset'] = X_test.reshape((X_test.shape[0],-1) )
                


            ## Initiate classification models
            for i in range(args.n_clients):
                if args.model == 'fedavg':
                    clients[i]['model'] = benchmark_models.digit_model_fedavg()
                elif args.model == 'fedbn':
                    clients[i]['model'] = benchmark_models.digit_model_fedbn()
                elif args.model == 'fedisk':
                    clients[i]['model'] = benchmark_models.digit_model_fedDisk()
                #vae model
                clients[i]['vae_model'] = CVAE(args.vae_latent_dim)
                clients[i]['vae_global_model'] = CVAE(args.vae_latent_dim)
                #MADE model
                clients[i]['made_model'] = made_model(hidden_units = made_hidden_neurons,hidden_layers = made_hidden_layers,learning_rate = 0.001,dropout = 0.1)
                clients[i]['made_model_global'] = made_model(hidden_units = made_hidden_neurons,hidden_layers = made_hidden_layers,learning_rate = 0.001,dropout = 0.1)
                

            ## ---------------------Train VAEs -------------------------------------

            ##support function
            
            if args.model=='fedisk' :
                if args.correction_method == 'vae_weight':
                    print("**USING VAE WEIGHTING")
                # Training vae locally
                    for client in clients:
                        print("Train VAE for client holding: {} dataset".format(client['dataset']))
                        vae_training(args.vae_iters, client, model_name='vae_model')
                    
                    ## Training global VAE using FL 
                    print("Train global VAE.")
                    for i in range(args.vae_iters+30):
                        for client in clients:
                            vae_training(1, client, model_name='vae_global_model')
                        w_cl = []
                        for num in range(args.n_clients):
                            w_cl.append(np.asarray(clients[num]['vae_global_model'].get_weights()) )
                        mean = np.mean(np.array(w_cl),axis=0)
                        for client in clients:
                            client['vae_global_model'].set_weights(mean)
                
            ###------------   Train a logistic classifier to get the sample weights---------------
                    for client in clients:  
                        mean, logvar = client['vae_model'].encode(client['X_train'])
                        z_local = client['vae_model'].reparameterize(mean, logvar) 
                        z_local_label = np.zeros(len(z_local))
                        mean, logvar = client['vae_global_model'].encode(client['X_train'])
                        z_global = client['vae_global_model'].reparameterize(mean, logvar)
                        z_global_label = np.ones(len(z_global) )
                        big_z  = np.concatenate( (z_local,z_global), axis=0 )
                        big_z_label = tf.keras.utils.to_categorical(np.concatenate((z_local_label , z_global_label ), axis=0 ))
                        clf = benchmark_models.fnn_model(args.shallow_neuron, (args.vae_latent_dim,)) 
                        clf.fit(big_z, big_z_label,  shuffle=True,epochs=shallow_iters,batch_size=32, \
                            verbose=2,validation_split = 0.1, callbacks=[callback] )
                        client['sample_weights'] =  normalize(clf.predict(z_local)[:,1])
                        print("20 Sample weight samples: {}".format(client['sample_weights'][0:20]) )
                        CVAE.generate_and_save_images(client['vae_model'], args.vae_iters, client['X_train'][0:16], log_dir=figures_dir+'/Samples', \
                            title='vae_local_{}'.format(client['dataset']))
                        CVAE.generate_and_save_images(client['vae_global_model'], args.vae_iters, client['X_train'][0:16], log_dir=figures_dir+'/Samples', \
                            title='vae_global_{}'.format(client['dataset']))
                
                
                ##------- train MADE model to estimate density globaly---------
                elif args.correction_method=='density_weight':
                    print("**USING DENSITY WEIGHTING")
                    learning_rate = 0.001
                    early_stopping = tf.keras.callbacks.EarlyStopping('val_loss', min_delta=0.1, patience=5)
                    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.001 * learning_rate)

                    ##train made locally
                    for client in clients:
                        if args.model_path:  
                            print("Loading density models from {}".format(args.model_path))
                            client['made_model'] = tf.keras.models.load_model('../Logs/{}/Models/made_model_{}'.format(args.model_path,client['dataset']))
                        else:
                            print("Training MADE for client holding: {} dataset".format(client['dataset']))
                            client['made_model'].fit(client['made_train_dataset'], client['made_train_dataset'],\
                                    epochs=made_iters, batch_size= made_batch, verbose=2,
                                    callbacks=[ early_stopping, reduce_lr],validation_data=(client['made_test_dataset'], client['made_test_dataset']),)
                        
                        yhat = client['made_model'].predict(client['made_test_dataset'])
                        display_digits(yhat, n=7, title=figures_dir+"/MADE_{}".format(client['dataset']))
                        client['made_model'].save(models_dir + '/made_model_{}'.format(client['dataset']))
                    
                    if  not args.model_path: 
                        for i in range(made_iters):
                            for client in clients:
                                history =  client['made_model_global'].fit(client['made_train_dataset'], client['made_train_dataset'],\
                                    epochs=1, batch_size= made_batch, verbose=2)
                            # Agregating models' weights 
                            w_cl = []
                            for num in range(args.n_clients):
                                w_cl.append(np.asarray(clients[num]['made_model_global'].get_weights()) )
                            # get the mean of models'weights
                            mean = np.mean(np.array(w_cl),axis=0)
                            for client in clients:
                                client['made_model_global'].set_weights(mean)

                        #save all flobal model
                        for client in clients:
                            client['made_model_global'].save(models_dir + '/made_model_global_{}'.format(client['dataset']))
                    else:# load global model
                        for client in clients:
                            client['made_model_global']=tf.keras.models.load_model('../Logs/{}/Models/{}/made_model_global_{}'.format(args.model_path,client['dataset']))

                    for client in clients:
                        print("-Train shallow network to find sample weights for client {}".format(client['dataset']))
                        ## Sanity check images 
                        yhat = client['made_model_global'].predict(client['made_train_dataset'])
                        display_digits(yhat, n=7, title=figures_dir+"/MADE_Global_{}".format(client['dataset']))
                        #
                        z_local = client['made_model'].predict(client['made_train_dataset']) 
                        z_local_label = np.zeros(len(z_local))
                        z_global = client['made_model_global'].predict(client['made_train_dataset'])
                        z_global_label = np.ones(len(z_global) )
                        big_z  = np.concatenate( (z_local,z_global), axis=0 )
                        big_z_label = tf.keras.utils.to_categorical(np.concatenate((z_local_label , z_global_label ), axis=0 ))
                        #if using fnn
                        using_shallow_1d = True
                        if not using_shallow_1d:
                            clf = benchmark_models.fnn_model(args.shallow_neuron, (z_local.shape[1],)) 
                        else: 
                            clf = benchmark_models.shallow_model_1d(args.shallow_neuron)
                            big_z = big_z.reshape((-1,28,28,1))
                        clf.fit(big_z, big_z_label,  shuffle=True,epochs=shallow_iters,batch_size=32, \
                            verbose=2,validation_split = 0.1, callbacks=[callback] )
                        #get output for train set
                        z_train = client['made_model'].predict(client['made_train_dataset']) 
                        if using_shallow_1d: z_train = z_train.reshape((-1,28,28,1))
                        p = clf.predict(z_train)[:,1]
                        client['sample_weights'] =  normalize(p/(1-p))
                        print("20 Sample weight samples: {}".format(client['sample_weights'][0:20]) )
                elif args.correction_method=='data_weight':
                    print("---Train Correction Network (Data weighting)---")
                    for k in range(args.n_clients):
                        big_X = np.concatenate( [clients[j]['X_train'] for j in range(args.n_clients)  ])
                        big_y = np.concatenate( [ np.ones(len(clients[j]['X_train'])) for j in range(args.n_clients)] )
                        # reduce the number of samples (Px) to be equavalent to client's samples (Qx)  
                        _, big_X, _, big_y = train_test_split(big_X, big_y, test_size= 1/args.n_clients , random_state=42)
                        shallow_model = benchmark_models.shallow_model(args.shallow_neuron)
                        combine_X = np.concatenate( (clients[k]['X_train'], big_X))
                        combine_y = np.concatenate( (np.zeros(len(clients[k]['y_train'])), np.ones(len( big_y) ) ) )
                        combine_y = tf.keras.utils.to_categorical( combine_y, num_classes=2 )
                        print("Local Training X shape : {}  y shape {}".format(clients[k]['X_train'].shape,clients[k]['y_train'].shape))
                        print("Big Training X shape : {}  y shape {}".format(big_X.shape,big_y.shape))
                        print("combine_X shape : {}  combine_y shape {}".format(combine_X.shape,combine_y.shape))
                        shallow_model.fit( combine_X, combine_y, shuffle=True,epochs=shallow_iters,batch_size=32, \
                            verbose=2,validation_split = 0.15, callbacks=[callback] )
                        p = shallow_model.predict(clients[k]['X_train'])[:,1]
                        clients[k]['sample_weights'] =  normalize( p/(1-p))
                        # clients[k]['sample_weights'] =  shallow_model.predict(clients[k]['X_train'])
                        print("50 Sample weight samples: {}".format(clients[k]['sample_weights'][0:20]) )
                        print("----")


            ## ----------------------Train FL models--------------------
            print('-Training FL Classifier ')
            for i in range(args.iters):
                for client in clients:
                    history =  client['model'].fit(client['X_train'], to_categorical(client['y_train'],n_classes),\
                        validation_split = 0.1,sample_weight= client['sample_weights'], epochs=1, batch_size= args.batch, verbose=2)
                    client['histories'].append(history.history)

                # Agregating models' weights 
                w_cl = []
                for num in range(args.n_clients):
                    w_cl.append(np.asarray(clients[num]['model'].get_weights()) )
                
                # get the mean of models'weights
                mean = np.mean(np.array(w_cl),axis=0)

                for client in clients:
                    client['model'].set_weights(mean)

            print("\n--Result for TRIAL {}/{}--".format(trial,args.n_trials))
            avg_acc = 0
            for c, client in enumerate(clients):
                client['acc'].append( client['model'].evaluate(client['X_test'],to_categorical(client['y_test'],n_classes), verbose=0)[1]*100 )
                print("Client {}   Acc: {:.2f}".format(client["dataset"].ljust(13), client['acc'][-1]) )
                avg_acc += client['acc'][-1]
                trial_results[client["dataset"]].append(client['acc'][-1]) # store results to global trials
            print("Average Acc: {:.2f}".format(avg_acc/args.n_clients))

        
        ##final report 
        ## ----------------------Testing on each client-----------------------------
        # log experiment parameters 
        print("--------Experiment Parameters-----")
        print("***Note : {}".format(args.note))
        print("- Correction method: {}".format(args.correction_method))
        print("- Datasets        : {}".format(datasets))
        print("- Model           : {}".format(args.model))
        print("- n_trials        : {}".format(args.n_trials))
        print("- Seed            : {}".format(args.seed))
        print("- Logist MaxIters : {}".format(args.logistic_iter))
        print("- N0 of Clients   : {}".format(args.n_clients))
        print("- Global Iteration: {}".format(args.iters))
        print("- VAE iteration   : {}".format(args.vae_iters))
        print("- VAE Latent Dim  : {}".format(args.vae_latent_dim))
        print("- MADE iteration  : {}".format(made_iters))
        print("- MADE layers     : {}".format(made_hidden_layers))
        print("- MADE neurons    : {}".format(made_hidden_neurons))
        print("- MADE Batch      : {}".format(made_batch))
        print("- Shallow neuron  : {}".format(args.shallow_neuron))
        print("- Shallow iters   : {}".format(shallow_iters))
        
        print("\n----------FINAL AVERAGE ACC OVER {} TRIALS-----------".format(args.n_trials))
        avg=[]
        for i in range(args.n_clients):
            avg.append( np.mean(np.array(trial_results[datasets[i]]) ) )
            print("Client {}   Acc: {:.2f}".format(datasets[i].ljust(13), avg[-1]) )
        final_avg = np.mean(np.array(avg))
        print("Average Acc: {:.2f}".format(final_avg ) )
        final_results.append( (final_avg, param) )
    
    
    print("--------------Final result including tuning params------------")
    print("(Acc, (made_iters, made_hidden_neurons, made_hidden_layers, made_batch, shallow_iters ))")
    final_results.sort(key=lambda result:result[0])
    print("Sorted: \n{}".format(str(final_results) ))

    ## dump clients data to pickle file
    # remove model in clients to be able to pickle
    for client in clients:
        try:
            client['model']            =None
            client['vae_model']        =None
            client['vae_global_model'] =None
            client['vae_train_dataset']=None
            client['vae_test_dataset'] =None
            client['made_model_global'] = client['made_model'] = None
            client['made_test_dataset'] = client['made_train_dataset'] =None 
        except:
            pass
        
    with open(log_dir+ '/clients.pkl','wb') as f:
        pickle.dump(clients, f)
    #plot loss function 
    plot_loss(log_dir)


    
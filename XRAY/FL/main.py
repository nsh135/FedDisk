""" Main file to train and test federated learning """

from gc import callbacks
from itertools import combinations
from tkinter import Y
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
import random, glob

# for reproducing hashbased algorigsm 
os.environ['PYTHONHASHSEED'] = '0'

datasets = [ 'ChinaSet','chest', 'lung' ]

n_classes = 2
image_size = (32,32)
row=column=image_size[0]

# prepare Logs directory 
now = datetime.now()
dt = now.strftime("%d%m%Y_%H:%M:%S")
log_dir = '../Logs/{}'.format(dt)
logfile = log_dir + '/'+ 'log.log'
figures_dir= log_dir + '/Figures'
models_dir = log_dir + '/Models'
#data path 

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

if not os.path.exists(log_dir):
  os.makedirs(log_dir)
if not os.path.exists(log_dir):
  os.makedirs(models_dir)
if not os.path.exists(figures_dir):
  os.makedirs(figures_dir+'/Samples')


def prepare_data(args):
    """
    Append Data for each client 
    """
    print("Loading datasets...")
    datasets_max_size = {'ChinaSet':660, 'chest': 5200, 'lung': 16000 } #  corresponding to [ 'ChinaSet','chest', 'lung' ]
    if args.n_clients == 3:
        per_clt_size = 500
        datasets_max_size = {'ChinaSet':per_clt_size, 'chest': per_clt_size, 'lung': per_clt_size } # 500, 500, 500
    elif args.n_clients == 10:
        per_clt_size = 500
        datasets_max_size = {'ChinaSet':per_clt_size * 1, 'chest': per_clt_size * 4, 'lung': per_clt_size * 5 }  # 500, 2000, 2500
    elif args.n_clients == 20:
        per_clt_size = 300
        datasets_max_size = {'ChinaSet':per_clt_size * 2, 'chest': per_clt_size * 8, 'lung': per_clt_size * 10 }
    elif args.n_clients == 50:
        per_clt_size = 300
        datasets_max_size = {'ChinaSet':per_clt_size * 2, 'chest': per_clt_size * 15, 'lung': per_clt_size * 33 } # 600, 4500, 9900
    else:
        raise Exception("Sorry, number of clients must be one of these: 3, 10, 50")
    
    # initiate clients
    clients =[]
    client_ID = 0
    for dataset in datasets:
        if dataset == 'ChinaSet':
            ChinaPath = '../DATA_XRAY/ChinaSet_AllFiles/CXR_png/*.png'
            image_list = []
            labels = []
            for id,filename in enumerate(glob.glob(ChinaPath)): #assuming gif
                im=Image.open(filename)
                im = im.resize(image_size)
                im = np.asarray(im)/255
                im = 1-im #flip black-white to white-black
                if len(im.shape)==3: im = np.mean(im,axis=-1)
                l = filename.split('_')[-1][0]
                labels.append(int(l=='1'))
                image_list.append(im)
                if id >= datasets_max_size[dataset]: break

        elif dataset=='chest':
            normal = '../DATA_XRAY/chest_xray/train/NORMAL/*.jpeg'
            abnormal = '../DATA_XRAY/chest_xray/train/PNEUMONIA/*.jpeg'
            image_list = []
            labels = []
            for l,path in enumerate([normal,abnormal]):
                for id,filename in enumerate(glob.glob(path)): #assuming gif
                    im=Image.open(filename)
                    im = im.resize(image_size)
                    im = np.asarray(im)/255
                    if len(im.shape)==3: im = np.mean(im,axis=-1)
                    labels.append(l)
                    image_list.append(im)
                    if id >= datasets_max_size[dataset]: break
        elif dataset=='lung':
            normal = '../DATA_XRAY/COVID-19_Radiography_Dataset/Normal/*.png'
            abnormal = '../DATA_XRAY/COVID-19_Radiography_Dataset/Lung_Opacity/*.png'
            image_list = []
            labels = []
            for l,path in enumerate([normal,abnormal]):
                for id,filename in enumerate(glob.glob(path)): #assuming gif
                    im=Image.open(filename)
                    im = im.resize(image_size)
                    im = np.asarray(im)/255
                    if len(im.shape)==3: im = np.mean(im,axis=-1)
                    labels.append(l)
                    image_list.append(im)
                    if id >= datasets_max_size[dataset]: break

        #resize images and scale (/255)
        X = np.array(image_list)
        X = X.reshape((-1, image_size[0],image_size[1],1 ))
        y = np.array(labels)
        #perturb data
        X , _ , y , _ = train_test_split(X,y,test_size=0.001,random_state=42)

        #convert to grey
        
        x_recover = np.rint(X[0,:,:,0]*255).astype(np.uint8)
        pil_image1=Image.fromarray(x_recover, mode='L')
        pil_image1.save(figures_dir + '/Samples/img_after_resized_{}_{}.jpg'.format(dataset, y[0]))

        
        
        # assign dataset to corresponding clients
        partition_start_idx = 0
        while partition_start_idx < datasets_max_size[dataset]:
            # permutation 
            X_train, X_test, y_train, y_test = \
                train_test_split(X[partition_start_idx:partition_start_idx+per_clt_size], \
                                y[partition_start_idx:partition_start_idx+per_clt_size], test_size=0.2, random_state=42)

            clients.append(
                {'client_ID' : client_ID,\
                'dataset':dataset, \
                'X_train': X_train, \
                'y_train': y_train, \
                'X_test': X_test,\
                'y_test': y_test ,\
                'histories':[],\
                'acc':[], \
                'sample_weights' : None  } )
            #update partition_start_idx
            partition_start_idx += per_clt_size
            client_ID += 1

            print("Client {} (0-index) :".format(clients[-1]['client_ID']))
            print("Dataset loaded: {}".format(dataset))
            print("X_train.shape: {}".format( X_train.shape) )
            print("y_train.shape: {}".format( y_train.shape) )
            print("X_test.shape: {}".format(X_test.shape))
            print("y_test.shape: {}".format( y_test.shape))
            # print("Samples: {}:{}".format( X_train[0], y_train[0]))
    print("Finish loading data to clinents!")
    return clients


def normalize(x):
    """ normalize array x to [0,1]"""
    if (np.max(x) - np.min(x)) != 0:
        r = (x-np.min(x))/(np.max(x) - np.min(x))*2
    else:
        r=x/x
    print("**sample clf output (p) and weights normalized normalized:{} \nmean r:{} std r:{}".format(x[:10], np.mean(r[:10]) , np.std(r) ))
    return r


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  default='fedisk',choices= ['fedbn','fedavg','fedisk'],  help = 'method to use, ')
    parser.add_argument('--seed', type = int, default= 13, help ='seed')
    parser.add_argument('--n_trials', type = int, default= 3, help ='number of trials')
    parser.add_argument('--params_search', type = bool, default= False, help ='test  with the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--n_clients', type = int, default= 3, help ='number of clients among 3,10,50')
    parser.add_argument('--global_iters', type = int, default=15, help = 'global training iterations')
    parser.add_argument('--made_iters', type = int, default=300, help = 'MADE iterations')
    parser.add_argument('--made_hidden_neurons', type = int, default=2000, help = 'number of neuron per layer for MADE model')
    parser.add_argument('--made_hidden_layers', type = int, default=2, help = 'number of hidden layers in MADE model')
    parser.add_argument('--made_batch', type = int, default= 64, help ='batch size for training made')
    parser.add_argument('--logistic_iter', type = int, default=50, help = 'max logistic regression iterations')
    parser.add_argument('--shallow_neuron', type = int, default= 100, help ='number of neuron for shallow network (adversarial training for calculating weight)')
    parser.add_argument('--shallow_iters', type = int, default= 20, help ='number of iteration for the adversarial training')
    parser.add_argument('--note',  default= '', help ='A note will be written down in the log')
    parser.add_argument('--correction_method',  default= 'MADE_weight', choices= ['data_weight','MADE_weight'])
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
        trial_results=[]
        for i in range(args.n_clients):
            trial_results.append([])
            trial_results[-1] = []

        for trial in range(args.n_trials):
            print("\n----------TRIAL {}/{}-----------".format(trial,args.n_trials))
            # prepare data and Clients dictionary to store variables
            clients = prepare_data(args) ## list of clients , each contains a dict of datacat out and model
            for client in clients:
                #  assign data to train density made model
                if not toGrey: # e.g., dim = 64x64x3
                    client['made_train_dataset'] = np.mean(client['X_train'], axis=-1).reshape((client['X_train'].shape[0],-1) )
                    client['made_test_dataset'] = np.mean(client['X_test'], axis=-1).reshape((client['X_test'].shape[0],-1) )
                else:# e.g. dim = 64x64x1
                    client['made_train_dataset'] = client['X_train'].reshape(( client['X_train'].shape[0],-1) )
                    client['made_test_dataset'] = client['X_test'].reshape(( client['X_test'].shape[0],-1) )
                

            ## Initiate classification models
            for i in range(args.n_clients):
                if args.model == 'fedavg':
                    clients[i]['model'] = benchmark_models.digit_model_fedavg()
                elif args.model == 'fedbn':
                    clients[i]['model'] = benchmark_models.digit_model_fedbn()
                elif args.model == 'fedisk':
                    clients[i]['model'] = benchmark_models.digit_model_fedDisk()
                #MADE model
                clients[i]['made_model'] = made_model(hidden_units = made_hidden_neurons,hidden_layers = made_hidden_layers,learning_rate = 0.001,dropout = 0.1)
                clients[i]['made_model_global'] = made_model(hidden_units = made_hidden_neurons,hidden_layers = made_hidden_layers,learning_rate = 0.001,dropout = 0.1)
                

            # ## ---------------------Train Auxilary model -------------------------------------
            
            if args.model=='fedisk' : 
                ##------- train MADE model to estimate density globaly---------
                if args.correction_method=='MADE_weight':
                    print("** DENSITY WEIGHTING (Using MADE model)")
                    learning_rate = 0.001
                    early_stopping = tf.keras.callbacks.EarlyStopping('val_loss', min_delta=0.1, patience=5)
                    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.001 * learning_rate)
                    callbacks = [ early_stopping, reduce_lr]

                    ## train made locally
                    for client in clients:
                        if args.model_path:  
                            print("Loading density models from {}".format(args.model_path))
                            client['made_model'] = tf.keras.models.load_model('../Logs/{}/Models/made_model_client_{}'.format(args.model_path,client['client_ID']))
                        else:
                            print("Training MADE for client {} holding: {} dataset".format(client['client_ID'], client['dataset']))
                            client['made_model'].fit(client['made_train_dataset'], client['made_train_dataset'],\
                                    epochs=made_iters, batch_size= made_batch, verbose=2,
                                    callbacks=callbacks,validation_data=(client['made_test_dataset'], client['made_test_dataset']),)
                        
                        yhat = client['made_model'].predict(client['made_test_dataset'])
                        display_digits(yhat, n=7, title=figures_dir+"/MADE_client_{}".format(client['client_ID']))
                        client['made_model'].save(models_dir + '/made_model_client_{}'.format(client['client_ID']))
                    
                    ## clients joinly train global MADE model and aggregating
                    if  not args.model_path:   
                        avg_global_val_loss_over_iters = [] #over made_iters
                        for i in range(made_iters):
                            #check if Made global become overfiting 
                            if len(avg_global_val_loss_over_iters)> 5 \
                                and avg_global_val_loss_over_iters[-3] < avg_global_val_loss_over_iters[-2] < avg_global_val_loss_over_iters[-1] :
                                print("MADE val loss greater than previous step causing overfitting. \nBreak training after {} global iteration. ".format(i))
                                break
                            avg_global_val_loss_each_iter = []
                            for client in clients:
                                history =  client['made_model_global'].fit(client['made_train_dataset'], client['made_train_dataset'],\
                                    epochs=1, batch_size= made_batch, verbose=2, validation_data=(client['made_test_dataset'], client['made_test_dataset']))
                                avg_global_val_loss_each_iter.append(history.history['val_loss'])
                            mean_loss = np.mean(np.array(avg_global_val_loss_each_iter))
                            print("Average val loss of global made over clients:{}".format(mean_loss))
                            avg_global_val_loss_over_iters.append(mean_loss)
                            # Agregating models' weights 
                            w_cl = []
                            for num in range(args.n_clients):
                                w_cl.append(np.asarray(clients[num]['made_model_global'].get_weights()) )
                            # get the mean of models'weights
                            mean = np.mean(np.array(w_cl),axis=0)
                            for client in clients:
                                client['made_model_global'].set_weights(mean)

                        #save all global model
                        for client in clients:
                            client['made_model_global'].save(models_dir + '/made_model_global_{}'.format(client['client_ID']))
                    else:# load global model
                        for client in clients:
                            client['made_model_global']=tf.keras.models.load_model('../Logs/{}/Models/{}/made_model_global_{}'.format(args.model_path,client['client_ID']))

                    for client in clients:
                        print("-Train shallow network to find sample weights for client {}".format(client['client_ID']))
                        ## Sanity check images 
                        yhat = client['made_model_global'].predict(client['made_train_dataset'])
                        display_digits(yhat, n=7, title=figures_dir+"/MADE_Global_{}".format(client['client_ID']))
                        #
                        z_local = client['made_model'].predict(client['made_train_dataset']) 
                        z_local_label = np.zeros(len(z_local))
                        z_global = client['made_model_global'].predict(client['made_train_dataset'])
                        z_global_label = np.ones(len(z_global) )
                        big_z  = np.concatenate( (z_local,z_global), axis=0 )
                        big_z_label = np.concatenate((z_local_label , z_global_label ))
                        #if using fnn
                        using_shallow_fnn = False
                        if  using_shallow_fnn:
                            clf = benchmark_models.fnn_model(args.shallow_neuron, (z_local.shape[1],)) 
                        else: #using cnn
                            clf = benchmark_models.shallow_model(args.shallow_neuron)
                            big_z = big_z.reshape((-1,row,column,1))
                        clf.fit(big_z, big_z_label,  shuffle=True,epochs=shallow_iters,batch_size=32, \
                            verbose=2,validation_split = 0.1, callbacks=[callback] )
                        #get output for train set
                        z_train = client['made_model'].predict(client['made_train_dataset']) 
                        if not using_shallow_fnn: z_train = z_train.reshape((-1,row,column,1))
                        p = clf.predict(z_train)[:0]
                        client['sample_weights'] =  1 -abs(0.5-p)*2 
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
                        print("Local Training X shape : {}  y shape {}".format(clients[k]['X_train'].shape,clients[k]['y_train'].shape))
                        print("Big Training X shape : {}  y shape {}".format(big_X.shape,big_y.shape))
                        print("combine_X shape : {}  combine_y shape {}".format(combine_X.shape,combine_y.shape))
                        shallow_model.fit( combine_X, combine_y, shuffle=True,epochs=shallow_iters,batch_size=32, \
                            verbose=2,validation_split = 0.15, callbacks=[callback] )
                        p = shallow_model.predict(clients[k]['X_train'])
                        #print("***DEBUG: output prediction from shallow network: {}".format(shallow_model.predict(clients[k]['X_train'])))
                        clients[k]['sample_weights'] = 1 -abs(0.5-p)*2 
                        print("20 Sample weight samples: {}".format(clients[k]['sample_weights'][0:20]) )
                        print("----")


            ## ----------------------Train FL models--------------------
            print('-Training FL Classifier ')
            mean_val_loss_over_inters = []
            for i in range(args.global_iters):
                if i>5 and mean_val_loss_over_inters[-3] < mean_val_loss_over_inters[-2] < mean_val_loss_over_inters[-1]:
                    print("Federated Learning becomes overfitting with average val_loss over iters: {} ".format(mean_val_loss_over_inters))
                    print("Break training at iteration {}".format(i))
                    break 
                val_loss_over_clients = []
                for client in clients:
                    history =  client['model']\
                        .fit(client['X_train'], to_categorical(client['y_train'],n_classes),\
                            validation_split = 0.1,sample_weight= client['sample_weights'], \
                            epochs=1, batch_size= args.batch, verbose=2,\
                            validation_data=(client['X_test'], to_categorical(client['y_test'],n_classes)) )
                    client['histories'].append(history.history)
                    val_loss_over_clients.append(history.history['val_loss'] )
                mean_val_loss = np.mean(np.array(val_loss_over_clients))
                mean_val_loss_over_inters.append(mean_val_loss)
                print("Mean Val Loss over clients: {}".format(mean_val_loss))
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
                print("Client {}, dataset {}   Acc: {:.2f}".format(client['client_ID'],client["dataset"].ljust(13), client['acc'][-1]) )
                avg_acc += client['acc'][-1]
                trial_results[client["client_ID"]].append(client['acc'][-1]) # store results to global trials
            print("Average Acc: {:.2f}".format(avg_acc/args.n_clients))

        
        ##final report 
        ## ----------------------Testing on each client-----------------------------
        # log experiment parameters 
        print("--------Experiment Parameters-----")
        print("***Note : {}".format(args.note))
        print("- Correction method: {}".format(args.correction_method if args.model=="fedisk" else "NA"))
        print("- Datasets        : {}".format(datasets))
        print("- Model           : {}".format(args.model))
        print("- n_clients        : {}".format(args.n_clients))
        print("- n_trials        : {}".format(args.n_trials))
        print("- Seed            : {}".format(args.seed))
        print("- Logist MaxIters : {}".format(args.logistic_iter))
        print("- N0 of Clients   : {}".format(args.n_clients))
        print("- Global Iteration: {}".format(args.global_iters))
        print("- MADE iteration  : {}".format(made_iters))
        print("- MADE layers     : {}".format(made_hidden_layers))
        print("- MADE neurons    : {}".format(made_hidden_neurons))
        print("- MADE Batch      : {}".format(made_batch))
        print("- Shallow neuron  : {}".format(args.shallow_neuron))
        print("- Shallow iters   : {}".format(shallow_iters))
        
        print("\n----------FINAL AVERAGE ACC OVER {} TRIALS-----------".format(args.n_trials))
        avg=[]
        for client in clients:
            avg.append( np.mean(np.array(trial_results[client['client_ID']]) ) )
            print("Client {}   Acc: {:.2f}".format(client['client_ID'], avg[-1]) )
        final_avg = np.mean(np.array(avg))
        print("Average Acc: {:.2f} ".format(final_avg) )
        final_results.append( (final_avg, param) )
    
    
    print("--------------Final result including tuning params------------")
    print("(Avg_Acc, Std, (made_iters, made_hidden_neurons, made_hidden_layers, made_batch, shallow_iters ))")
    final_results.sort(key=lambda result:result[0])
    print("Sorted: \n{}".format(str(final_results) ))

    ## dump clients data to pickle file
    # remove model in clients to be able to pickle
    for client in clients:
        try:
            client['model']            = None
            client['made_model_global'] = client['made_model'] = None
            client['made_test_dataset'] = client['made_train_dataset'] =None 
        except:
            pass
        
    with open(log_dir+ '/clients.pkl','wb') as f:
        pickle.dump(clients, f)
    #plot loss function 
    plot_loss(log_dir)


    
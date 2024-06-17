import os
import warnings
import socket
from tqdm.auto import tqdm

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


import CONST
from transforms import load_transform
from config.defaults import default_argument_parser, cfg_costum_setup, get_run_id, create_tensorboard_run_dict
from datasets import load_dataset
from engine.loops import sample_dataset, train_vanilla, validate
from engine.checkpoints import save_model
from models import load_model, load_freezed_model
from optimizers import load_optimizer
from utils.utils_files import better_hparams
from evaluation import evaluate


warnings.filterwarnings("ignore", message=r"The frame.append", category=FutureWarning)

logs_dir = CONST.STORAGE_DIR

def train(cfg):

    print("Train Config:\n", cfg)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    hostname = socket.getfqdn()
    run_id = get_run_id()
    basename = "{}_{}".format(cfg.TRAIN.DATASET, cfg.MODEL.NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device.type)


    if device.type == 'cuda':
        print("CUDA INFO:")
        print("Number of GPUs available:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA devices:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        is_gpu = True
    else:
        is_gpu = False

    # LOAD DATA AUGMENTATION/TRANSFORMATION METHOD ----- ????????
    if cfg.AUG.PROB > 0:
        insize = cfg.TRAIN.INPUT_SIZE_IRR
        input_transform = load_transform()
    else:
        input_transform = None

    ds = load_dataset(ds_name=cfg.TRAIN.DATASET,
                      input_transform=input_transform,
                      input_size=cfg.TRAIN.INPUT_SIZE_IRR,
                      num_classes=cfg.TRAIN.NUM_CLASSES,
                      num_kpts=cfg.TRAIN.NUM_KEYPOINTS,
                      allowed_kpts=cfg.TRAIN.ALLOWED_KEYPOINTS)

    #data loader
    trainloader, valloader, testloader = sample_dataset(trainset=ds.trainset,
                                                valset=ds.valset,
                                                testset=ds.testset,
                                                overfit=cfg.TRAIN.OVERFIT,
                                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                                num_workers=cfg.DATA_LOADER.NUM_WORKERS)

    # --------------------------------------
    # TEST: check labels from the cycle classifier

    min_label = float('inf')
    max_label = float('-inf')

    for images, targets, paths in trainloader:
        for t in targets:

            # Update min_label and max_label based on the current batch
            batch_min_label = torch.min(t["labels"]).item()
            batch_max_label = torch.max(t["labels"]).item()

            min_label = min(min_label, batch_min_label)
            max_label = max(max_label, batch_max_label)

    print("Minimum cycle Label:", min_label)
    print("Maximum cycle Label:", max_label)

    # -------------------

    model = load_model(cfg,is_gpu=is_gpu) # notice the default num of keypooints
    print("Model loaded. Output type:", model.output_type)

    if cfg.TRAIN.CHECKPOINT_FILE_PATH is not None:
        print("Loading pre-trained model:", cfg.TRAIN.CHECKPOINT_FILE_PATH)
        model = load_freezed_model(cfg.TRAIN.CHECKPOINT_FILE_PATH)

    model.to(device)
    print('training model {}..'.format(model.__class__.__name__))

    # Load weights ????????????????????????????

    if (cfg.TRAIN.WEIGHTS is not None) and (os.path.exists(cfg.TRAIN.WEIGHTS)):
         print("loading weights {}..".format(cfg.TRAIN.WEIGHTS))
         checkpoint = torch.load(cfg.TRAIN.WEIGHTS)
         model.load_state_dict(checkpoint['model_state_dict'])
    # Set training parameters:
    class_weights = torch.Tensor([1]*(1 + cfg.NUM_FRAMES));  class_weights[0] = 0.1;   class_weights /= len(class_weights)
    class_weights = class_weights.to(device)

    # Loss and optimizer:

    if len(cfg.MODEL.LOSS_FUNC) == 1:
        loss = cfg.MODEL.LOSS_FUNC[0]
    else:
        loss = cfg.MODEL.LOSS_FUNC #workaround for handling multiple losses in cfg 
    #criterion = load_loss(loss, device=device, class_weights=class_weights) # SERGI: class_weights in EchoNet but not in CycleDetect
    criterion = torch.nn.MSELoss()

    optimizer = load_optimizer(method_name=cfg.SOLVER.OPTIMIZER, parameters=model.parameters(), learningrate=cfg.SOLVER.BASE_LR)
    #print(dict(model.named_parameters()))

    # Logger setup:
    log_folder = os.path.join(logs_dir, 'logs', cfg.TRAIN.DATASET,
                                cfg.MODEL.NAME, str(cfg.MODEL.BACKBONE), run_id)
    
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    print("Saving logs in:", log_folder)

    # CycleDetect metrics
    best_val_MAP = 0
    best_val_metric = {"kpts": np.inf, "ef": np.inf, "sd": np.inf, "bxs":np.inf}
    #evaluator = ObjectDetectEvaluator(dataset=ds.valset, output_dir=None, verbose=False)

    # KptDetector metrics
    best_val_loss = np.inf
    best_val_metric = {"kpts": np.inf, "ef": np.inf, "sd": np.inf}
    metric_dict = {'BestVal/kptsErr': 1, 'BestVal/efErr': 1, 'BestVal/sdErr': 1}
    #evaluator =  DopplerEvaluator(dataset=ds.valset, output_dir=None, verbose=False) if cfg.TRAIN.DATASET == 'doppler' else EchonetEvaluator(dataset=ds.valset, tasks=["kpts"], output_dir=None, verbose=False)

    writer = SummaryWriter(log_dir=log_folder)
    run_dict = create_tensorboard_run_dict(cfg)
    run_dict["hostname"] = hostname
    
    sei = better_hparams(writer, hparam_dict=run_dict, metric_dict=metric_dict)


    with open(os.path.join(log_folder,"train_config.yaml"), "w") as f:
        f.write(cfg.dump())   # save config to file

    # Train & Evaluate
    print("Training in batches of size {}..".format(cfg.TRAIN.BATCH_SIZE))
    print('Training on machine name {}..'.format(hostname))
    print("Using data augmentation type {} for {:.2f}% of the input data".format(cfg.AUG.METHOD, 100 * cfg.AUG.PROB))

    with tqdm(total=cfg.TRAIN.EPOCHS) as pbar_main:
        for epoch in range(1, cfg.TRAIN.EPOCHS+1):
            pbar_main.update()

            train_losses = train_vanilla(epoch=epoch,
                                       loader=trainloader,
                                       optimizer=optimizer,
                                       model=model,
                                       device=device,
                                       criterion=criterion,
                                       prossesID=run_id)
            train_loss = train_losses["main"].avg
            # Add losses to the tensorboard writer
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('TrainLoss/Loss', train_loss, epoch)
            writer.add_scalar('TrainLoss/BBox', train_losses["bbox_loss"].avg, epoch) # BY NOW IT COMES FROM BBOX REG, NOT ABSOLUTE BBOX ERROR
            writer.add_scalar('TrainLoss/Class', train_losses["class_loss"].avg, epoch)
            writer.add_scalar('TrainLoss/RPN', train_losses["rpn_loss"].avg, epoch)
            writer.add_scalar('TrainLoss/Object', train_losses["object_loss"].avg, epoch)
            
            print("EPOCH {}:".format(epoch), train_losses)
            filename = os.path.join(log_folder, 'epoch_{}_weights_{}_best_{}Err_{}.pth'.format(epoch, basename, 'kpts', 0))
            save_model(filename, epoch, model, cfg, train_loss, 0, 0, hostname)
            #print(torch.cuda.memory_stats(device=device))

            """
            # eval: ------------------------------------------------------------------------------------ ################################


            
            if epoch % cfg.TRAIN.EVAL_INTERVAL == 0:
                val_losses, val_outputs, val_inputs = validate(mode='validation',
                                                               epoch=epoch,
                                                               loader=valloader,
                                                               model=model,
                                                               device=device,
                                                               criterion=criterion,
                                                               prossesID=run_id)
                val_loss = val_losses["main"].avg
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                writer.add_scalar('ValLoss/Loss', val_losses["kpt_loss"], epoch)

                evaluator.process(val_inputs, val_outputs)
                eval_metrics = evaluator.evaluate()


                #-----------------------------------
                # STORE BEST KPT DETECTOR:
                if val_loss < best_val_loss:
                    filename = os.path.join(log_folder, 'weights_{}_best_loss.pth'.format(basename))
                    best_val_loss = val_loss
                    save_model(filename, epoch, model, args, train_loss, val_loss, best_val_metric, hostname)
                    print("Saved at loss {:.5f}\n".format(val_loss))
                    writer.add_scalar('BestVal/Loss', best_val_loss, epoch)

                # Update best val metric:
                for task in ['ef', 'sd', 'kpts']:
                    if task in eval_metrics:
                        metric = eval_metrics[task]['norm'] if task == 'kpts' else eval_metrics[task] #DOPPLER
                        if  metric < best_val_metric[task]:
                            filename = os.path.join(log_folder, 'weights_{}_best_{}Err.pth'.format(basename, task))
                            best_val_metric[task] = metric
                            writer.add_scalar("BestVal/{}Err".format(task), best_val_metric[task], epoch)
                            if task in ['ef', 'kpts']:
                                save_model(filename, epoch, model, cfg, train_loss, val_loss, best_val_metric, hostname)
                                print("Saved at val loss {:.5f}, {} error {:.5f}%\n".format(val_loss, task, metric))
                if epoch % 100 == 0:
                    filename = os.path.join(log_folder, 'epoch_{}_weights_{}_best_{}Err_{}.pth'.format(epoch, basename, 'kpts', eval_metrics['kpts']['norm']))
                    save_model(filename, epoch, model, cfg, train_loss, val_loss, best_val_metric, hostname)
                    print("Saved at val loss {:.5f}, {} error {:.5f}%\n".format(val_loss, 'kpts', eval_metrics['kpts']['norm']))

            """
            # --------------------------------------------------------------------------################################################
            if epoch % cfg.TRAIN.EVAL_INTERVAL == 0:
                errors_by_label, mean_error = evaluate(model, device)
                print("EVALUATION MEAN ERROR:", mean_error)
                writer.add_scalar('Validation kpt loss', mean_error, epoch)
 
    # Save & Close:
    print('Finished Training')
    filename = os.path.join(log_folder, 'weights_{}_ep_{}.pth'.format(basename, epoch))
    save_model(filename, epoch, model, cfg, train_loss, 1, 1, hostname) #)val_loss, best_val_metric, hostname)
    writer.file_writer.add_summary(sei)
    writer.close()  # close tensorboard
    return train_loss


if __name__ == '__main__':

    args = default_argument_parser()
    cfg = cfg_costum_setup(args)
    train(cfg)
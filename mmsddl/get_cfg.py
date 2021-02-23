import argparse
import platform
import torch
from nutsflow.config import Config
from pathlib import Path
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
gtc = ['Tonic-clonic']
fbtc = ['FBTC']
focal_myoclonic = ['focal,Myoclonic']
automatisms = ['Automatisms']
epileptic_spasms = ['Epileptic spasms']
gnr_tonic = ['gnr,Tonic']
behaviour_arrast = ['Behavior arrest']
szr_all = ['szr']
'''


def get_CFG():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    if platform.system() == 'Linux':
        parser.add_argument('-r', '--ROOT', default='/fast2/',
                            help='root path for input data')
        parser.add_argument('-o', '--out_dir',
                            default='/slow1/out_datasets/bch/full_data_single_out',
                            help='path to output prediction')
    elif platform.system() == 'Darwin':
        parser.add_argument('-r', '--ROOT',
                            default=os.path.join(Path.home(), 'datasets',
                                                 'bch'),
                            help='root path for input data')
        parser.add_argument('-o', '--out_dir',
                            default='/Users/jbtang/datasets/bch/full_data_single_out',
                            help='path to output prediction')
    else:
        print('Unknown OS platform %s' % platform.system())
        exit()

    parser.add_argument('--DATADIR',
                        default='wristband_REDCap_202102_szr_cluster_win0')

    parser.add_argument('--n_epochs', default=4, type=int,
                        help='epochs')
    parser.add_argument('--szr_types', default='gnr,Tonic-clonic', type=str,
                        help='szr_types')
    parser.add_argument('--modalities', default='EDA,ACC', type=str,
                        help='szr_types')

    parser.add_argument('--lr', default=1e-3, type=float,
                        help='learning rate')
    parser.add_argument('--batch_size', default=32, type=int,  # 100
                        help='mini-batch size (default: 256)')

    parser.add_argument('--win_len', default=10, type=int,
                        help='epochs')
    parser.add_argument('--win_step', default=2, type=int,
                        help='epochs')

    parser.add_argument('--preictal_len', default=60, type=int,
                        help='preictal_len')
    parser.add_argument('--postictal_len', default=60, type=int,
                        help='postictal_len')
    parser.add_argument('--motor_threshold', default=0.1, type=float,
                        help='motor_threshold')

    parser.add_argument('--sing_wrst', default=1, type=int,
                        help='0: two wrst when possible, 1: single wrst only')

    args = parser.parse_args()

    args.modalities = [item for item in args.modalities.split(',')]

    modality_path = ''
    for modality in sorted(args.modalities):
        modality_path += modality

    # middle_path = 'redcap_results/' + args.szr_types + '_' + modality_path + '_win_' + str(
    #     args.win_len) + '_step_' + str(args.win_step) + '_sing_wrst_' + str(
    #     args.sing_wrst)

    middle_path = 'redcap_results/' + args.szr_types + '_' + modality_path + '_win_' + str(
        args.win_len) + '_step_' + str(args.win_step)

    '''
    Feb 22, 2021 Stats:
                                         
    Semiology	Patients	Number of Patients	unique	non-unique
    Total	 	91	513	863
    focal | motor | FBTC	{'c226', 'c421', 'c423', 'c189', 'c356', 'c433', 'c192', 'c241', 'c232', 'c242', 'c429', 'c417', 'c234', 'c225', 'c303', 'c245', 'c399', 'c299', 'c392', 'c296', 'c388'}	21	38	59
    gnr | motor | Tonic	{'c147', 'c326', 'c340', 'c404', 'c243', 'c378', 'c353', 'c196', 'c313', 'c372', 'c364', 'c236', 'c370', 'c212'}	14	88	163
    focal | motor | Tonic	{'c356', 'c263', 'c328', 'c388', 'c325', 'c242', 'c261', 'c329', 'c278', 'c377', 'c399', 'c390', 'c296', 'c235'}	14	44	67
    focal | subclinical | invalid	{'c263', 'c190', 'c282', 'c390', 'c365', 'c391', 'c425', 'c278', 'c283', 'c411', 'c305', 'c123', 'c274'}	13	61	83
    focal | motor | Automatisms	{'c195', 'c284', 'c316', 'c221', 'c190', 'c427', 'c396', 'c389', 'c418', 'c391', 'c235'}	11	26	42
    focal | non-motor | Behavior arrest	{'c422', 'c190', 'c328', 'c282', 'c365', 'c394', 'c389', 'c329', 'c403', 'c390', 'c303'}	11	21	34
    gnr | motor | Epileptic spasms	{'c428', 'c147', 'c285', 'c432', 'c410', 'c196', 'c273', 'c406'}	8	47	88
    gnr | motor | Tonic-clonic	{'c380', 'c333', 'c290', 'c387', 'c372', 'c309'}	6	15	26

    '''

    if args.szr_types == 'FBTC':
        patients = ['c226', 'c421', 'c423', 'c189', 'c356', 'c433', 'c192',
                    'c241', 'c232', 'c242', 'c429', 'c417', 'c234', 'c225',
                    'c303', 'c245', 'c399', 'c299', 'c392', 'c296', 'c388']
    elif args.szr_types == 'gnr,Tonic':
        patients = ['c147', 'c326', 'c340', 'c404', 'c243', 'c378', 'c353',
                    'c196', 'c313', 'c372', 'c364', 'c236', 'c370', 'c212']
    elif args.szr_types == 'focal,Tonic':
        patients = ['c356', 'c263', 'c328', 'c388', 'c325', 'c242', 'c261',
                    'c329', 'c278', 'c377', 'c399', 'c390', 'c296', 'c235']
    elif args.szr_types == 'focal,subclinical':
        patients = ['c263', 'c190', 'c282', 'c390', 'c365', 'c391', 'c425',
                    'c278', 'c283', 'c411', 'c305', 'c123', 'c274']
    elif args.szr_types == 'focal,Automatisms':
        patients = ['c195', 'c284', 'c316', 'c221', 'c190', 'c427', 'c396',
                    'c389', 'c418', 'c391', 'c235']
    elif args.szr_types == 'focal,Behavior arrest':
        patients = ['c422', 'c190', 'c328', 'c282', 'c365', 'c394', 'c389',
                    'c329', 'c403', 'c390', 'c303']
    elif args.szr_types == 'gnr,Epileptic spasms':
        patients = ['c428', 'c147', 'c285', 'c432', 'c410', 'c196', 'c273',
                    'c406']
    elif args.szr_types == 'gnr,Tonic-clonic':
        patients = ['c380', 'c333', 'c290', 'c387', 'c372', 'c309']
    elif args.szr_types == 'szr':
        patients = None
    else:
        print('not supported seizure type', args.szr_types)
        patients = None

    if patients is not None:
        patients = [p.upper() for p in sorted(patients)]

    args.szr_types = [args.szr_types]

    if args.sing_wrst == 0:
        sing_wrst = False
    else:
        sing_wrst = True

    CFG = Config(
        patients=patients,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        verbose=1,
        win_len=args.win_len,
        win_step=args.win_step,
        rootdir=args.ROOT,
        datadir=os.path.join(args.ROOT, args.DATADIR),
        traincachedir=os.path.join(args.ROOT, middle_path, 'cache/train/fold'),
        valcachedir=os.path.join(args.ROOT, middle_path, 'cache/val/fold'),
        testcachedir=os.path.join(args.ROOT, middle_path, 'cache/test/fold'),
        plotdir=os.path.join(args.ROOT, middle_path, 'plots'),
        ckpdir=os.path.join(args.ROOT, middle_path, 'checkpoints'),
        metric_results_dir = os.path.join(args.ROOT,'redcap_results/'),
        modalities=args.modalities,
        szr_types=args.szr_types,
        preictal_len=args.preictal_len,
        postictal_len=args.postictal_len,
        motor_threshold=args.motor_threshold,
        sequence_model=False,
        cacheclear=True,
        sing_wrst=sing_wrst,
        )

    return CFG

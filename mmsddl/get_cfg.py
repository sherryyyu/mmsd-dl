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
    elif platform.system() == 'Darwin':
        parser.add_argument('-r', '--ROOT',
                            default=os.path.join(Path.home(), 'datasets',
                                                 'bch'),
                            help='root path for input data')
    else:
        print('Unknown OS platform %s' % platform.system())
        exit()

    parser.add_argument('--DATADIR',
                        default='wristband_REDCap_202102_szr_cluster_win0')

    parser.add_argument('--n_epochs', default=4, type=int,
                        help='epochs')
    parser.add_argument('--szr_types', default='gnr,Tonic-clonic', type=str,
                        help='szr_types')
    parser.add_argument('--modalities', default='EDA', type=str,
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

    parser.add_argument('--max_cpu', default=1, type=int,
                        help='parallel tasks')

    parser.add_argument('--crossfold', default=-1, type=int,
                        help='-1: LOO, otherwise number of folds')

    parser.add_argument('--results_dir',type=str,
                        default='redcap_results',
                        help='path to output prediction')

    args = parser.parse_args()

    results_dir = os.path.join(args.ROOT, args.results_dir)


    args.modalities = [item for item in args.modalities.split(',')]
    args.modalities = sorted(args.modalities)

    modality_path = ''
    for modality in args.modalities:
        modality_path += modality

    # middle_path = 'redcap_results/' + args.szr_types + '_' + modality_path + '_win_' + str(
    #     args.win_len) + '_step_' + str(args.win_step) + '_sing_wrst_' + str(
    #     args.sing_wrst)

    middle_path = os.path.join(args.results_dir, args.szr_types + '_' + modality_path + '_win_' + str(
        args.win_len) + '_step_' + str(args.win_step))

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
        patients = ['c189', 'c192', 'c225', 'c226', 'c232', 'c234', 'c241', 'c242', 'c245', 'c296',
                    'c299', 'c303', 'c356', 'c388', 'c392', 'c399', 'c417', 'c421', 'c423', 'c429', 'c433']
    elif args.szr_types == 'focal,Tonic':
        patients = ['c235', 'c242', 'c261', 'c263', 'c278', 'c296', 'c325', 'c328', 'c329', 'c356',
                    'c366', 'c377', 'c388', 'c390', 'c399']
    elif args.szr_types == 'gnr,Tonic':
        patients = ['c147', 'c196', 'c212', 'c236', 'c243', 'c313', 'c326', 'c330', 'c340', 'c353',
                    'c364', 'c370', 'c372', 'c378', 'c404']
    elif args.szr_types == 'focal,subclinical':
        patients = ['c123', 'c190', 'c263', 'c274', 'c278', 'c282', 'c283', 'c305', 'c358', 'c365',
                    'c390', 'c391', 'c411', 'c425']
    elif args.szr_types == 'focal,Automatisms':
        patients = ['c190', 'c195', 'c221', 'c235', 'c284', 'c316', 'c389', 'c391', 'c396', 'c418', 'c427']
    elif args.szr_types == 'focal,Behavior arrest':
        patients = ['c190', 'c282', 'c303', 'c328', 'c329', 'c365', 'c389', 'c390', 'c394', 'c403', 'c422']
    elif args.szr_types == 'gnr,Epileptic spasms':
        patients = ['c147', 'c196', 'c273', 'c285', 'c406', 'c410', 'c428', 'c432']
    elif args.szr_types == 'focal,Clonic':
        patients = ['c212', 'c226', 'c232', 'c358', 'c427', 'c429']
    elif args.szr_types == 'gnr,Tonic-clonic':
        patients = ['c290', 'c309', 'c333', 'c372', 'c380', 'c387']
    elif args.szr_types == 'szr':
        patients = ['c123', 'c147', 'c189', 'c190', 'c192', 'c195', 'c196', 'c197', 'c198', 'c200',
                    'c212', 'c213', 'c218', 'c221', 'c225', 'c226', 'c228', 'c232', 'c234', 'c235',
                    'c236', 'c241', 'c242', 'c243', 'c245', 'c261', 'c263', 'c269', 'c273', 'c274',
                    'c278', 'c282', 'c283', 'c284', 'c285', 'c290', 'c296', 'c299', 'c300', 'c302',
                    'c303', 'c305', 'c308', 'c309', 'c313', 'c316', 'c325', 'c326', 'c328', 'c329',
                    'c330', 'c333', 'c336', 'c340', 'c353', 'c356', 'c358', 'c362', 'c364', 'c365',
                    'c366', 'c369', 'c370', 'c372', 'c377', 'c378', 'c379', 'c380', 'c387', 'c388',
                    'c389', 'c390', 'c391', 'c392', 'c394', 'c396', 'c399', 'c403', 'c404', 'c406',
                    'c410', 'c411', 'c417', 'c418', 'c421', 'c422', 'c423', 'c425', 'c427', 'c428',
                    'c429', 'c430', 'c432', 'c433']
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
        metric_results_dir = results_dir,
        modalities=args.modalities,
        szr_types=args.szr_types,
        preictal_len=args.preictal_len,
        postictal_len=args.postictal_len,
        motor_threshold=args.motor_threshold,
        sequence_model=False,
        cacheclear=True,
        sing_wrst=sing_wrst,
        max_cpu=args.max_cpu,
        crossfold = args.crossfold
        )

    return CFG

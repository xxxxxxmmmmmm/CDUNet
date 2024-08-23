import os
import argparse
from torch.backends import cudnn
from loader import get_loader
from solver import Solver
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

    solver = Solver(args, data_loader)
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    #train为训练模式，test为测试模式
    parser.add_argument('--mode', type=str, default='train')
    #parser.add_argument('--mode', type=str, default='test')

    parser.add_argument('--load_mode', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='E:\B30_3mm')
    parser.add_argument('--saved_path', type=str, default='./npy_B30/')
    parser.add_argument('--save_path', type=str, default='./CDUNet/')
    parser.add_argument('--test_patient', type=str, default='L506')
    parser.add_argument('--result_fig', type=bool, default=True)

    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)
    # parser.add_argument('--trunc_min', type=float, default=0)
    # parser.add_argument('--trunc_max', type=float, default=255)
    parser.add_argument('--transform', type=bool, default=False)

    # if patch training, batch size is (--patch_n * --batch_size)
    #16 64 8
    parser.add_argument('--patch_n', type=int, default=3)
    #方便切块训练，扩充一圈
    #scunet为136，scunet2为128,ODCNet为136
    parser.add_argument('--patch_size', type=int, default=136)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--print_iters', type=int, default=20)
    parser.add_argument('--decay_iters', type=int, default=3000)
    parser.add_argument('--save_iters', type=int, default=1000)

    parser.add_argument('--test_iters', type=int, default=53000)

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--multi_gpu', type=bool, default=False)

    args = parser.parse_args()
    main(args)

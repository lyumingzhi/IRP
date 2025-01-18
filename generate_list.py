import os
import argparse
import random
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,default='clean')    
parser.add_argument('--path', type=str,default='/home2/huangyi/ImageNet100/train')                                                
args = parser.parse_args()

folders = sorted(os.listdir(args.path))
if not os.path.exists("list"):
    os.makedirs("list")

fl = open('list/' + args.name + '_list.txt', 'w')
fn = open('list/' + args.name + '_name.txt', 'w')

# sub_folders = os.listdir('/home1/huangyi/ImageNet-100/train')

# random.shuffle(folders)
# classes = ['n01697457', 'n01847000', 'n01728920', 'n01496331', 'n01664065', 'n01751748', 'n01744401', 'n01704323', 'n01675722', 'n01742172']

num=0
for i, folder in enumerate(folders):
    # if folder in classes:
        # continue
    # else:
    #     num+=1
    # if num==200:
    #     break

    fn.write(str(i) + ' ' + folder + '\n')

    folder_path = os.path.join(args.path, folder)
    files = os.listdir(folder_path)

    for file in files:

        fl.write('{} {}\n'.format(os.path.join(folder_path, file), i))

fl.close()
fn.close()

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --job-name=download
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


cd /vast/hvp2011/data/
mkdir -p domainnet
cd ./domainnet

wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip


wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip



wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip






# wget http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt


# wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt

# wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt
# wget http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt
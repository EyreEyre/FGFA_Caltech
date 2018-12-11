
import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, 'fgfa_rfcn'))

import train_end2end
import test
import demo
import demo_0

if __name__ == '__main__':
	# train_end2end.main()
	# test.main()
	# demo.main()
	demo_0.main()
"""单 realization 裸自谱缓存 worker（v3 口径）。"""
import sys
sys.path.insert(0, '/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/code')
import run_envclass_fullspec_mcmc as m
real, cosmo = int(sys.argv[1]), sys.argv[2]
m.build_cache(('v', 's', 'f'), cosmo, [real])

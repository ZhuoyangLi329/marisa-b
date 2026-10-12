"""单 realization knot 裸自谱缓存 worker（v3 口径，与 v/s/f 缓存同款）。"""
import sys
sys.path.insert(0, '/pscratch/sd/l/lzy/reconstuction-png-v2/02_reconstruction_on_pk/code')
import run_envclass_fullspec_mcmc as m
real, cosmo = int(sys.argv[1]), sys.argv[2]
m.build_cache(('k',), cosmo, [real])

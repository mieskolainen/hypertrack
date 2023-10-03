# hypertrack

from . import iotools

__version__    = '0.0.1'
__release__    = 'alpha'
__repository__ = 'github.com/mieskolainen/hypertrack'
__author__     = 'm.mieskolainen@imperial.ac.uk'

print(f'hypertrack | ver {__version__} ({__repository__}) | {__author__}')
print(iotools.sysinfo())

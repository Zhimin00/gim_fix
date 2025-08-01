import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
SPIDER_REPO_PATH = path.normpath(path.join(HERE_PATH, '../../spider'))
SPIDER_LIB_PATH = path.join(SPIDER_REPO_PATH, 'spider')
# check the presence of models directory in repo to be sure its cloned
if path.isdir(SPIDER_LIB_PATH):
    # workaround for sibling import
    sys.path.insert(0, SPIDER_REPO_PATH)
else:
    raise ImportError(f"dust3r is not initialized, could not find: {SPIDER_LIB_PATH}.\n "
                      "Did you forget to run 'git submodule update --init --recursive' ?")
import os.path as osp
import shutil

from imitation.helpers.logger import info, debug, set_level, configure, logkv, dumpkvs


DEBUG = 10


def test_logger():
    info("hi")
    debug("shouldn't appear")
    set_level(DEBUG)
    debug("should appear")
    dir = "/tmp/testlogging"
    if osp.exists(dir):
        shutil.rmtree(dir)
    configure(dir_=dir)
    logkv("a", 3)
    logkv("b", 2.5)
    dumpkvs()
    logkv("b", -2.5)
    logkv("a", 5.5)
    dumpkvs()
    info("^^^ should see a = 5.5")

    logkv("b", -2.5)
    dumpkvs()

    logkv("a", "longasslongasslongasslongasslongasslongassvalue")
    dumpkvs()


if __name__ == "__main__":
    test_logger()

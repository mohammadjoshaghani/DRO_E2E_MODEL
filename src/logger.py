import sys, os
import logging, datetime

# some directory to save loggs
if not os.path.exists(os.getcwd() + "/log/"):
    os.makedirs(os.getcwd() + "/log/")

# makes logger
now = datetime.datetime.now().strftime("%d-%b-%y_%H-%M-%S")

logging.basicConfig(
    filename=f"log/{now}_log.txt",
    filemode="w",
    format="%(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

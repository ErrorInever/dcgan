import argparse
import logging


def parse_args():
    parser = argparse.ArgumentParser(description='dcGAN')
    pass


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    ch = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
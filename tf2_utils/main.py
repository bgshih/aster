import argparse
import os
from typing import List

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

from tf2_utils.inferer import Inferer

TARGET_IMAGE_HEIGHT = 64
TARGET_IMAGE_WIDTH = 256
CHAR_VECTOR = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


def infere_images(images_path: List[str]):
    inferer = Inferer()
    tokenizer = get_tokenizer()
    for image_path in images_path:
        image = load_image(image_path)
        logits = inferer(image)
        sequence_length = [logits.shape[1]]
        sequences_decoded = tf.nn.ctc_greedy_decoder(
            tf.transpose(logits, [1, 0, 2]), sequence_length, merge_repeated=False
        )[0][0]
        sequences_decoded = tf.sparse.to_dense(sequences_decoded).numpy()

        word = tokenizer.sequences_to_texts(sequences_decoded)[0]
        print(word)

def get_tokenizer():
    tokenizer = Tokenizer(char_level=True, lower=False, oov_token="<OOV>")
    tokenizer.fit_on_texts(CHAR_VECTOR)
    return tokenizer

def load_image(image_path:str):
    image = cv2.imread(os.path.join(image_path))

    image = cv2.resize(
        image, (TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT)
    )
    image = image.astype(np.float32) / 127.5 - 1.0

    return tf.expand_dims(tf.constant(image), 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_path",
        nargs="+",
        type=str,
        help="Path to the images to infere",
    )
    args = parser.parse_args()
    infere_images(args.images_path)

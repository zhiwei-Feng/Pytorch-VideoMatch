import os
from random import shuffle
from itertools import combinations

from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import sys

sys.path.append(".")
from log import logger


class Davis(Dataset):
    modes = 'train', 'val', 'trainval'

    def __init__(self, base_dir, mode='train', use_seq=None):
        super().__init__()

        assert (mode in self.modes)

        self._base_dir = base_dir
        self._annotations_dir = os.path.join(self._base_dir, "Annotations", "480p")
        self._images_dir = os.path.join(self._base_dir, "JPEGImages", "480p")
        self._image_sets_dir = os.path.join(self._base_dir, "ImageSets", "480p")

        alldirs = self._annotations_dir, self._images_dir, self._image_sets_dir
        checkdirs = [os.path.isdir(d) for d in alldirs]
        if not all(checkdirs):
            raise ValueError("This is not expected DAVIS dataset dir structure, "
                             "you need the following dirs: \n\t{}".format("\n\t".join(alldirs)))

        self.mode = [mode] if mode in self.modes[:-1] else self.modes[:-1]

        self.seq_names = []

        for m in self.mode:
            filepath = os.path.join(self._image_sets_dir, m + ".txt")
            with open(filepath) as file:
                # self.seq_names.extend(s.strip() for s in file.readlines())
                for s in file.readlines():
                    # s: /JPEGImages/480p/bear/00000.jpg /Annotations/480p/bear/00000.png
                    name = s.strip().split()[0].split("/")[3]  # ex: bear
                    self.seq_names.append(name)

        # davis datasets contain duplicates
        self.seq_names = set(self.seq_names)

        if use_seq is not None:
            # check if specified sequences are valid
            use_seq = set(use_seq)
            if not use_seq.issubset(self.seq_names):
                raise RuntimeError("Specified set of sequence names isn't subset of loaded DAVIS dataset (year: 2016),"
                                   "\ngiven: {},\nvalid: {}".format(use_seq, self.seq_names))

            # only use specified sequences
            self.seq_names = use_seq & self.seq_names

        self.sequences = [Sequence(name, self._annotations_dir, self._images_dir) for name in sorted(self.seq_names)]

        logger.debug("Number of specified sequences in davis dataset is {}".format(self.__len__()))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, img_pos):
        seq_idx, frame_idx = img_pos
        img_path, ann_path = self.sequences[seq_idx][frame_idx]

        # load image and annotation with PIL
        img = Image.open(img_path).convert("RGB")
        ann = Image.open(ann_path).convert("L")

        frame = Frame(self.sequences[seq_idx], frame_idx, img, ann)

        return frame


class Frame:
    def __init__(self, seq, frame_idx, img, ann):
        self.seq = seq
        self.frame_idx = frame_idx
        self.img = img
        self.ann = ann


class Sequence:
    def __init__(self, name, base_ann_dir, base_img_dir):
        self.name = name
        self._ann_dir = os.path.join(base_ann_dir, name)
        self._img_dir = os.path.join(base_img_dir, name)

        # print("name:{}, _ann_dir:{}, _img_dir:{}".format(self.name, self._ann_dir, self._img_dir))
        # console> name:bear, _ann_dir:./DAVIS\Annotations\480p\bear, _img_dir:./DAVIS\JPEGImages\480p\bear

        frame_imgs = [os.path.join(self._img_dir, frame_no) for frame_no in sorted(os.listdir(self._img_dir))]
        frame_anns = [os.path.join(self._ann_dir, frame_no) for frame_no in sorted(os.listdir(self._ann_dir))]

        assert (len(frame_imgs) == len(frame_anns))

        self.frames = [(img, ann) for img, ann in zip(frame_imgs, frame_anns)]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def __eq__(self, other):
        return other is not None and self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class PairSampler(Sampler):
    def __init__(self, dataset, randomize=True):
        super().__init__(data_source=dataset)
        self._dataset = dataset
        self._randomize = randomize
        self._sequences = self._dataset.sequences

        self._all_pairs = []
        for seq_idx, seq in enumerate(self._sequences):
            idx_pairs = [(seq_idx, frame_idx) for frame_idx in range(len(seq))]
            self._all_pairs.extend(list(combinations(idx_pairs, 2)))

        if self._randomize:
            shuffle(self._all_pairs)

        logger.debug("Number of all pairs {}".format(self.__len__()))

    def get_indexes(self, index_list):
        return np.array(self._all_pairs)[index_list].tolist()

    def __iter__(self):
        for pair_idx in self._all_pairs:
            yield pair_idx

    def __len__(self):
        return len(self._all_pairs)


def collate_pairs(data):
    ref_frame, test_frame = data

    return (ref_frame.img, ref_frame.ann), (test_frame.img, test_frame.ann)


class MultiFrameSampler(Sampler):
    def __init__(self, dataset):
        super().__init__(data_source=dataset)
        self._dataset = dataset
        self._sequences = self._dataset.sequences

        self._samples = []
        for seq_idx, seq in enumerate(self._sequences):
            idx_pairs = [(seq_idx, frame_idx) for frame_idx in range(len(seq))]
            self._samples.extend(idx_pairs)

    def __iter__(self):
        for frame_idx in self._samples:
            yield frame_idx

    def __len__(self):
        return len(self._samples)


def collate_multiframes(data):
    return data


if __name__ == '__main__':
    davis = Davis("./DAVIS")

    ps = PairSampler(davis)

    ps = iter(ps)
    pair_i = next(ps)
    print(pair_i)
    first_pair = davis[pair_i[0]], davis[pair_i[1]]
    ref_frame: Frame = first_pair[0]
    test_frame: Frame = first_pair[1]
    print(ref_frame.seq, ref_frame.frame_idx, ref_frame.img, ref_frame.ann)
    print("First frames shapes = ", ref_frame.img.size, ref_frame.ann.size)
    print("Second frames shapes = ", test_frame.img.size, test_frame.ann.size)

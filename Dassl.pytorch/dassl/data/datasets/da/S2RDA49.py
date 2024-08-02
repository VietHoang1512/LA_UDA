import os.path as osp
from dassl.utils import listdir_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class S2RDA49(DatasetBase):
    dataset_dir = "S2RDA-49"
    domains = ["synthetic", "real"]
    class_names =  [
                "airplane", "bag", "basket", "bathtub", "bed", "bench", "bicycle", "birdhouse", "bottle", "bowl", 
                "bus", "cabinet", "camera", "cap", "car", "chair", "clock", "dishwasher", "filecabinet", "flowerpot",
                "guitar", "helmet", "keyboard", "knife", "lamp", "laptop", "loudspeaker", "mailbox", "microphone", 
                "microwaves", "motorcycle", "mug", "piano", "pillow", "plant", "printer", "remote", "rifle", 
                "skateboard", "sofa", "stove", "table", "telephone", "tower", "train", "trashbin", "truck", "washer",
                "watercraft"
                ]

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        train_x = self._read_data("synthetic")
        train_u = self._read_data("real")
        test = self._read_data("real")

        super().__init__(train_x=train_x, train_u=train_u, test=test)

    def _read_data(self, dname):
        filedir = "train" if dname == "synthetic" else "validation"
        items = []
        # There is only one source domain
        domain = 0

        for label, class_name in enumerate(self.class_names):
            class_path = osp.join(self.dataset_dir, dname, class_name)
            print("Loading", class_path)
            imnames = listdir_nohidden(class_path)

            for imname in imnames:
                impath = osp.join(class_path, imname)
                item = Datum(
                    impath=impath,
                    label=label,
                    domain=domain,
                    classname=class_name.lower(),
                )
                items.append(item)
        return items 

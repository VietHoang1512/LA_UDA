import os.path as osp
from dassl.utils import listfiles_nohidden

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class S2RDAMS39(DatasetBase):
    dataset_dir = "S2RDA-MS-39"
    domains = ["synthetic", "real"]
    class_names =  [
                    "airplane", "bed", "bowl", "car", "helmet", "laptop", "mug", "remote", "table", "trashbin",
                    "bag", "bench", "bus", "chair", "keyboard", "loudspeaker", "pillow", "skateboard", "telephone", "truck",
                    "basket", "bicycle", "cabinet", "clock", "knife", "microwaves", "plant", "sofa", "tower", "watercraft",
                    "bathtub", "bottle", "camera", "dishwasher", "lamp", "motorcycle", "printer", "stove", "train"
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
            imnames = listfiles_nohidden(class_path)  # TODO: each S2RDAMS39 image folder contains itself, LOL

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

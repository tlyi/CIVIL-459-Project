import openpifpaf

from .openlane_kp import OpenLaneKp


def register():
    openpifpaf.DATAMODULES['openlane'] = OpenLaneKp
    # keep checkpoints for now
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-openlane-24'] = \
        "http://github.com/DuncanZauss/openpifpaf_assets/releases/" \
        "download/v0.1.0/shufflenetv2k16-201113-135121-apollo.pkl.epoch290"

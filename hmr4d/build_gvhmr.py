from omegaconf import OmegaConf
from hmr4d import PROJ_ROOT
from hydra.utils import instantiate
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL


def build_gvhmr_demo(ckpt_path=None):
    if ckpt_path is None:
        ckpt_path = PROJ_ROOT / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
    cfg = OmegaConf.load(PROJ_ROOT / "hmr4d/configs/demo_gvhmr_model/siga24_release.yaml")
    gvhmr_demo_pl: DemoPL = instantiate(cfg.model, _recursive_=False)
    gvhmr_demo_pl.load_pretrained_model(ckpt_path)
    return gvhmr_demo_pl.eval()

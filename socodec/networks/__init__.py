import os

from ..utils.config import Config
from ..utils.utils import module_search, load_checkpoint


def build_network(conf):
    modules = []
    
    for module_name, module_conf in conf.items():
        _trainable = module_conf.pop('_trainable', True)
        _ckpt = module_conf.pop('_ckpt', None)
        
        # Load configurations
        _conf = module_conf.pop('_conf', None)
        if _conf:
            loaded_conf = Config(_conf).task.network.get(module_name)
            module_conf.update(loaded_conf)
            _second_ckpt = module_conf.pop('_ckpt', None) # Remove _ckpt in the referred config file

        # Load class name
        _name = module_conf.pop('_name', None)
        module_class = module_search(_name, os.path.dirname(__file__), 'socodec.networks')
        module = module_class(**module_conf)

        # Load weights (optional)
        if _ckpt:
            load_checkpoint(module, _ckpt)

        if not _trainable:
            for param in module.parameters():
                param.requires_grad = False

        modules.append((module_name, module))

    return modules